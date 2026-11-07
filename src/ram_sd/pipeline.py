"""Four explicit stages; only Query (without a gold label) enters inference."""

from . import prompts
from .io import canonical
from .parsing import appendix_verdict, maintext_verdict, planner_output


def validate_plan(recommendation, text, enabled):
    plan = recommendation["selected_plan"]
    changes = []
    if enabled:
        if plan == "simple_irony" and len(text.split()) > 50:
            plan = "expectation_violation"
            changes.append("Appendix E.2: more than 50 whitespace-separated words")
        entities = {e.strip().casefold() for e in recommendation["knowledge_entities"]}
        if len(entities) >= 3:
            plan = "knowledge_dependent"
            changes.append("Appendix E.2: at least three distinct specific entities")
    return plan, changes


def agents_for(plan, config):
    if plan == "expectation_violation":
        return ("semantic", "expectation", "incongruity", "rhetoric")
    if plan == "knowledge_dependent":
        return ("semantic", "knowledge", "alignment", "rhetoric")
    if plan == "simple_irony":
        return ("semantic", "rhetoric") if config.simple_irony_semantic_support else ("rhetoric",)
    raise ValueError(f"Unknown plan: {plan}")


class Pipeline:
    def __init__(self, config, client, retriever=None):
        self.config = config
        self.client = client
        self.retriever = retriever
        self.last_trace = None

    def predict(self, query, dataset):
        event_start = len(self.client.events)
        trace = {"row_id": query.row_id, "profile": self.config.profile, "steps": [],
                 "retrieval_enabled": self.config.retrieves(dataset), "exemplars": []}
        self.last_trace = trace

        def call(stage, prompt, scope=None):
            trace["active_stage"] = stage
            output = self.client.chat(prompt, self.config, scope or f"{dataset}:{query.row_id}:{stage}")
            trace["steps"].append({"stage": stage, "output": output})
            return output

        # Stage 1: top-k from each training-label subset, one rationale per exemplar, then similarity analysis.
        exemplars = []
        if self.config.retrieves(dataset):
            if self.retriever is None:
                raise ValueError("Retrieval required but no matching index was supplied")
            trace["active_stage"] = "retrieval"
            exemplars = self.retriever.retrieve(query)
            trace["exemplars"] = exemplars
            for exemplar in exemplars:
                rationale = call("rationale", prompts.render(
                    prompts.RATIONALE, text=exemplar["text"], label=exemplar["label"],
                    category="sarcastic" if exemplar["label"] else "non-sarcastic"),
                    scope=f'{dataset}:rationale:{exemplar["row_id"]}')
                exemplar["rationale"] = rationale
            similarity = call("similarity", prompts.render(prompts.SIMILARITY, text=query.text,
                context_if_available=f'Context: "{query.context}"' if query.context else "",
                non_sarcastic=prompts.examples_text(exemplars, 0), sarcastic=prompts.examples_text(exemplars, 1)))
        else:
            # MUSTARD does not instantiate or invoke a retriever or an embedding client.
            similarity = "Stage 1 disabled for MUSTARD (Appendix A.1). Use supplied dialogue context."

        # Stage 2: retrieved evidence is available before selecting P* and producing O_plan.
        prompt = prompts.render(prompts.PLANNER, text=query.text, similarity=similarity,
            non_sarcastic=prompts.examples_text(exemplars, 0), sarcastic=prompts.examples_text(exemplars, 1))
        prompt += "\n\nSupplied dialogue context: " + canonical(query.context)
        if self.config.planner_rule_validation:
            prompt += ('\nSerialization extension: return only JSON, and also include "knowledge_entities": '
                       '[a list of the specific entities identified in feature analysis], '
                       'so the published rule validation can be executed. Use [] when none.')
        else:
            prompt += "\nReturn only the requested JSON object."
        recommendation = planner_output(call("planner", prompt), self.config.planner_rule_validation)
        plan, changes = validate_plan(recommendation, query.text, self.config.planner_rule_validation)
        planner_context = recommendation["contextual_analysis"]
        trace["planner"] = {**recommendation, "executed_plan": plan, "rule_overrides": changes}
        shared = prompts.grounding(query, planner_context, similarity, exemplars)

        # Stage 3: six specialist roles; helper context prompt inherited from the authors' code.
        outputs = {}
        context_analysis = planner_context
        if self.config.use_context_agent and plan != "simple_irony":
            context_analysis = call("context", prompts.render(prompts.CONTEXT, text=query.text,
                context=canonical({"dialogue": query.context, "planner_context": planner_context})) + shared)
            outputs["context"] = context_analysis
        for agent in agents_for(plan, self.config):
            values = {"text": query.text, "context": query.context, "planner_context": planner_context,
                      "context_analysis": context_analysis, "semantic": outputs.get("semantic", ""),
                      "knowledge": outputs.get("knowledge", ""), "expectation": outputs.get("expectation", ""),
                      "similarity": similarity}
            outputs[agent] = call(agent, prompts.render(prompts.AGENTS[agent], **values) + shared)
        trace["agents_executed"] = list(outputs)

        # Stage 4: keep Appendix label-confidence and main-text class probability distinct.
        values = {"text": query.text, "context": query.context, "similarity": similarity,
                  "planner_context": planner_context, "plan_type": plan, "agent_outputs": canonical(outputs)}
        if self.config.profile == "appendix":
            integrated = call("integrator", prompts.render(prompts.INTEGRATOR, **values) + shared)
            verdict = appendix_verdict(integrated)
        else:
            integrated = call("integrator", prompts.render(prompts.MAIN_TEXT_SUMMARY, **values) + shared)
            verdict = maintext_verdict(call("judger", prompts.render(prompts.MAIN_TEXT_JUDGER,
                                       **values, integrated=integrated) + shared))
        trace.update(verdict)
        trace["plan_type"] = plan
        trace["status"] = "ok"
        trace["active_stage"] = "complete"
        trace["requests"] = self.client.events[event_start:]
        return trace
