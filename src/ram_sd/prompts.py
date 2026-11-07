"""Prompt templates based on Appendix E of the RAM-SD paper.

Source: https://aclanthology.org/2026.acl-long.64.pdf
"""

from string import Template
from textwrap import dedent

from .io import canonical, digest

RATIONALE = """
Analyze why the following text is $category:
Text: "$text" Label: $label
Provide a concise rationale (2-3 sentences) explaining:
1. Key linguistic features (tone, style, rhetorical devices)
2. Contextual cues or knowledge dependencies
3. Why this is clearly $category
Focus on concrete evidence rather than abstract concepts.
"""

SIMILARITY = """
You are a similarity analysis expert. Compare the query text with retrieved exemplars to identify linguistic patterns and stylistic similarities.
**Query Text:** "$text" $context_if_available
**Retrieved NON-SARCASTIC Examples (Label 0) with Rationales:**
$non_sarcastic
**Retrieved SARCASTIC Examples (Label 1) with Rationales:**
$sarcastic
Analyze the query's similarity patterns:
**Similarity to NON-SARCASTIC Examples:**
- Shared features: [tone, directness, literal expression, etc.]
- Pattern alignment: [similar linguistic structures or rhetorical styles]
- Strength of similarity: [strong/moderate/weak]
**Similarity to SARCASTIC Examples:**
- Shared features: [irony markers, contradiction patterns, mocking tone, etc.]
- Pattern alignment: [similar ironic devices or expectation violations]
- Strength of similarity: [strong/moderate/weak]
**Comparative Assessment:**
- Primary similarity direction: [more similar to sarcastic/non-sarcastic/mixed]
- Key discriminative features: [2-3 critical differences from one category]
- Confidence level: [high/medium/low]
**Contextual Inference:** Based on retrieved examples and their rationales, infer:
- Likely situational context of the query
- Potential background knowledge required for interpretation
- Candidate sarcasm type if sarcastic: [expectation_violation/knowledge_dependent/simple_irony]
Return structured analysis focusing on concrete pattern matching with exemplars.
"""

PLANNER = """
As meta-planner, analyze query and select optimal reasoning strategy.
**Query:** "$text"
**Similarity Analysis from Stage 1:** $similarity
**Retrieved Examples Summary:**
- Non-sarcastic: $non_sarcastic
- Sarcastic: $sarcastic
Based on similarity patterns and query features, analyze:
**Feature Analysis:**
1. Contradiction level: [none/low/medium/high] - between literal/intended meaning
2. Exaggeration/irony present: [yes/no]
3. Emotional conflict: [yes/no] - positive words with negative intent
4. Context dependency: [low/medium/high]
5. Rhetorical devices: [list key devices if present]
6. Knowledge entities: [list SPECIFIC entities: named persons, events, orgs]
   Exclude: generic concepts, common terms, abstract ideas
**Plan Selection (choose ONE):**
- expectation_violation: For contradiction/irony, pragmatic violations
- knowledge_dependent: For SPECIFIC entities requiring background knowledge
- simple_irony: For short (fewer than 15 words), overt ironic expressions
**Contextual Analysis (O_plan):** Synthesize inferred context from similarity analysis and examples:
- Likely situational context: [inferred scenario/setting]
- Relevant background knowledge: [key info for understanding]
- Expectation baseline: [what would be normal expression in this context]
- Pragmatic interpretation hints: [guidance for downstream agents]
**Output:** {"selected_plan": "expectation_violation/knowledge_dependent/simple_irony", "confidence": 0-1, "contextual_analysis": "synthesized context and interpretation guidance", "reasoning": "why this plan based on similarity patterns and features"}
Priority: Weight similarity patterns heavily - if query strongly matches sarcastic/non-sarcastic exemplar patterns, factor this into plan selection.
"""

# This helper is explicitly referenced in E.3.2/E.3.3, but has no template in E.
# Reuse the authors' local _context_agent template instead of inventing a hidden one.
CONTEXT = """
As a context understanding expert, please analyze the contextual information of the text:
Text: "$text"
Additional Context: "$context"
Please analyze:
1. Explicit context (such as dialogue context, scene description)
2. Implicit context (social cultural background, common sense assumptions)
3. Key contextual elements
4. Importance of context for understanding the text (1-10)
Return precise, concise structured analysis in several sentences.
"""

AGENTS = {
    "semantic": """
As a semantic analysis expert, deeply analyze the semantics of the text:
Text: "$text" Context: "$context"
Analysis:
1. Literal meaning and deep meaning
2. Emotional tendency and intensity (positive/negative/neutral, intensity 1-10)
3. Emotional coloring of key vocabulary
4. Overall tone (formal/informal/teasing/serious, etc.)
Return precise, concise structured analysis in several sentences.
""",
    "expectation": """
Build expectation model using prior analysis:
Text: "$text"
Analysis from prior agents:
- Context analysis (from Context Agent): $context_analysis
- Semantic analysis (from Semantic Agent): $semantic
- Contextual grounding (from Meta-Planner): $planner_context
Build expectation model:
1. Normal expectation: What would be typical expression in this context
2. Actual vs. expected: Compare query expression with baseline expectation
3. Deviation analysis: Degree (1-10) and type (semantic/tonal/pragmatic)
4. Intentionality: Whether deviation appears deliberate for rhetorical effect
Reference Meta-Planner's expectation baseline for contextual grounding. Return structured analysis (3-4 sentences).
""",
    "knowledge": """
As a knowledge retrieval expert, analyze entities and concepts in the text:
Text: "$text" Context: $context_analysis
Please retrieve and analyze:
1. Key entities, people, events mentioned in the text
2. Common evaluations, stereotypes, public perceptions of these entities
3. Related background knowledge and common sense
4. Importance of this knowledge for understanding text sarcasm
Return precise, concise structured analysis in several sentences.
""",
    "alignment": """
Check alignment between text and background knowledge:
Text: "$text" Semantic analysis: $semantic
Background knowledge: $knowledge
Similarity patterns (from Stage 1): $similarity
Analyze:
1. Consistency: Does literal meaning align with background knowledge/common sense
2. Distortion detection: Deliberate misrepresentation or counter-factual claims
3. Inconsistency assessment: Degree (1-10) and nature (factual/evaluative/tonal)
4. Sarcastic intent: Whether inconsistency serves ironic/mocking purpose
Consider similarity patterns - if query matches known sarcastic patterns with knowledge contradictions, weight this evidence.
Return structured analysis (3-4 sentences).
""",
    "incongruity": """
Specifically detect and quantify inconsistencies:
Text: "$text" Expectation analysis: $expectation
Please detect:
1. Specific types of inconsistency (semantic, emotional, logical, common sense, etc.)
2. Quantification of inconsistency degree (1-10)
3. Whether inconsistency has sarcastic effect
4. Key inconsistency points location and manifestation
Return precise, concise structured analysis in several sentences.
""",
    "rhetoric": """
Identify if there is any rhetorical devices in the text:
Text: "$text" Context: "$context"
Focus on identifying:
1. Irony, exaggeration, understatement
2. Puns, rhetorical questions, interrogative sentences
3. Contrast, analogy
4. Contribution of these rhetorical devices to the expression
5. Do not think too much, these may be replies to other people on social media
6. Judge it as you are a normal linguist, not a sarcasm detector
Return precise, concise structured analysis in several sentences.
""",
}

INTEGRATOR = """
As sarcasm expert, synthesize all evidence and make final judgment:
**Query:** "$text"
**Similarity Analysis (Stage 1):** $similarity
**Meta-Planner Output (Stage 2):** Selected plan: $plan_type,
Contextual analysis: $planner_context
**Agent Outputs (Stage 3):** $agent_outputs
**Decision Framework:**
*Positive Indicators:* Contradiction literal/intended; rhetorical questions with mocking;
exaggeration/understatement; emotional incongruity; ironic praise; expectation violations;
strong similarity to sarcastic exemplars
*Negative Indicators:* Direct criticism without irony; genuine questions; consistent tone;
literal matches intent; strong similarity to non-sarcastic exemplars
*Synthesis Requirements:*
1. Weight similarity analysis heavily - if Stage 1 shows strong pattern match to sarcastic/non-sarcastic, prioritize this evidence
2. Evaluate coherence: Do multiple agents agree on sarcastic interpretation?
3. Assess intentionality: Are contradictions/devices deliberate rhetorical strategy?
4. Context alignment: Does judgment match Meta-Planner's contextual inference?
**STRICT Decision Logic:**
- Multiple clear indicators (irony+contradiction+mocking) + similarity to sarcastic -> sarcastic
- Strong similarity to sarcastic patterns + agent consensus on irony -> sarcastic
- Obvious contradiction literal/intended with intentionality -> sarcastic
- Direct criticism/opinions without ironic devices -> non-sarcastic
- Genuine inquiry (not mockery) + similarity to non-sarcastic -> non-sarcastic
- Emotional expression without contradiction + non-sarcastic patterns -> non-sarcastic
**Domain Context:** Forum posts/comments/replies. Apply STRICT standards - only classify sarcastic
with CLEAR evidence (intentional irony/mockery/contradiction). Casual criticism without ironic intent -> non-sarcastic.
**Output Format:** Line 1: <<LABEL>> 1 (sarcastic) or 0 (non-sarcastic)
Line 2: JSON {"label": 1/0, "conf": 0-1, "reasoning": "synthesis referencing similarity analysis, agent findings, and decision rationale"}
"""

# The paper does not publish these two main-text-mode templates. They are explicit reconstructions.
MAIN_TEXT_SUMMARY = """
Synthesize a coherent evidence summary for sarcasm detection, grounded only in the supplied analysis.
Query: "$text"
Dialogue context: $context
Similarity analysis: $similarity
Meta-Planner: $planner_context
Agent outputs: $agent_outputs
Identify supporting and conflicting evidence, distinguish literal criticism from intentional irony,
and state uncertainty. Return a concise evidence summary for a separate Judger.
"""
MAIN_TEXT_JUDGER = """
Make the final sarcasm judgment from the query and aggregated reasoning evidence.
Query: "$text"
Dialogue context: $context
Structured reasoning trace: $agent_outputs
Integrated evidence: $integrated
Return ONLY JSON {"p_sarcasm": number between 0 and 1, "explanation": "a concise explanation citing specific evidence"}.
p_sarcasm means the probability of label 1 (sarcastic), not confidence in whichever label is predicted.
"""


def render(template, **values):
    return Template(dedent(template).strip()).substitute(values)


def examples_text(exemplars, label):
    return "\n".join(f'{i}. Text: "{e["text"]}" | Rationale: {e["rationale"]}' for i, e in enumerate((e for e in exemplars if e["label"] == label), 1)) or "None (Stage 1 disabled)."


def grounding(query, planner_context, similarity, exemplars):
    return "\n\nShared grounding (Appendix E.3):\n" + canonical({
        "dialogue_context": query.context,
        "meta_planner_context": planner_context,
        "similarity_analysis": similarity,
        "rationale_augmented_exemplars": exemplars,
    })


def fingerprint():
    return digest({"rationale": RATIONALE, "similarity": SIMILARITY, "planner": PLANNER,
                   "context": CONTEXT, "agents": AGENTS, "integrator": INTEGRATOR,
                   "summary": MAIN_TEXT_SUMMARY, "judger": MAIN_TEXT_JUDGER,
                   "schema_extensions_version": 1})
