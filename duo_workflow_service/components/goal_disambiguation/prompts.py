PROMPT = """
### Input Format
Inputs should be provided within the following XML tags:

<task_description>
</task_description>

<conversation_history>
</conversation_history>

Note: Include all messages in chronological order within the conversation_history tags.

### Evaluation Criteria

1. IMMEDIATE TASK CLARITY
   - Are the core requirements explicitly stated?
   - Is the expected output format clearly defined?
   - Are there any ambiguous terms or phrases that need clarification?
   - Score (0-5): __

2. CONTEXTUAL CONSISTENCY
   - Does the task align with previously established context?
   - Are there any contradictions with earlier requirements?
   - Has relevant information from previous exchanges been considered?
   - Score (0-5): __

3. COMPLETENESS OF INFORMATION
   - Are all necessary parameters provided?
   - Are edge cases or special conditions specified?
   - Is there sufficient detail for implementation?
   - Score (0-5): __

4. TECHNICAL PRECISION
   - Are technical terms used correctly and consistently?
   - Are any required specifications (versions, formats, etc.) provided?
   - Are constraints and limitations clearly defined?
   - Score (0-5): __

### Analysis Process
1. First, carefully read the entire conversation history
2. Identify any relevant context or requirements established in previous exchanges
3. Analyze the current task description against this background
4. Evaluate each criterion independently
5. Consider potential ambiguities or missing information
6. Provide specific examples of unclear or missing elements
7. When Final Verdict is "NEEDS CLARIFICATION" or "UNCLEAR" with score below 4 use {clarification_tool} to provide suggestions for clarifying the task.

### Output Format

CLARITY ASSESSMENT REPORT

Overall Clarity Score: [Average of all criteria scores]

Strengths:
1. [Specific element that is well-defined]
2. [Another well-defined element]
[Add more as needed, using a numbered list]

Areas Needing Clarification:
1. [Specific ambiguity or missing information]
2. [Another area needing clarification]
[Add more as needed, using a numbered list]

Context Considerations:
1. [Relevant information from conversation history]
2. [Potential conflicts with previous requirements]
[Add more as needed, using a numbered list]

Recommendations:
1. [Specific suggestion for improving task clarity]
2. [Another suggestion for improvement]
[Add more as needed, using a numbered list]

Final Verdict:
[CLEAR/NEEDS CLARIFICATION/UNCLEAR]
[Brief explanation of the verdict in 2-3 sentences]

### Example Usage

<task_description>
Update the user profile page CSS
</task_description>

<conversation_history>
User: We need to modernize our UI. Can we start with the profile page?
Assistant: I can help with that. What specific aspects of the profile page would you like to update?
User: The whole styling needs to be refreshed.
</conversation_history>

Analysis:
- Immediate Task Clarity Score: 2/5 (Lacks specific styling requirements)
- Contextual Consistency Score: 4/5 (Aligns with previous UI discussions)
- Completeness Score: 1/5 (Missing responsive design requirements, browser compatibility needs)
- Technical Precision Score: 2/5 (No specific CSS properties or values mentioned)

Overall Score: 2.25/5

Recommendations:
1. Specify which elements need styling updates
2. Define target browser versions
3. Include responsive breakpoints
4. List specific CSS properties to modify
5. Provide desired values or ranges for each property

Verdict: NEEDS CLARIFICATION
The task lacks specific styling requirements and technical specifications needed for implementation.
While it aligns with previous discussions about UI updates, more detailed parameters are required for accurate execution.

### Instructions for Use
1. Replace placeholder text in [brackets] with actual content
2. Maintain consistent scoring across evaluations
3. Always consider both immediate context and conversation history
4. Provide actionable recommendations for improvement
5. Keep final verdicts concise but informative
"""

ASSIGNMENT_PROMPT = """
Now please assess clarity and completeness for:

<task_description>
{goal}
</task_description>

<conversation_history>
{conversation_history}
</conversation_history>
"""

SYS_PROMPT = """
You are an expert LLM Judge tasked with evaluating the clarity and precision of user task descriptions.
Your role is to analyze both the immediate task description
and the conversation history to ensure all necessary context is properly considered.
"""

CLARITY_JUDGE_RESPONSE_TEMPLATE = """
{response}{message}

I'm ready to help with your project but I need a few key details:

{recommendations}
"""
