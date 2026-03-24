setting openai key with powershell:
$env:OPENAI_API_KEY="your_real_key_here"

Reflection Questions


7. What is an LLM agent? 
An LLM agent is a system in which a language model goes beyond simply generating text and is able to take actions, use external tools, and perform multi-step reasoning. 
Instead of answering in one pass, it can decide what to do next, call tools such as a calculator or a knowledge base, observe the results, and then continue reasoning until it produces a final answer.

8. Why not always call tools? 
It is not always ideal to call tools because doing so introduces additional overhead. 
Tool usage can slow down responses, increase computational cost, and create more opportunities for errors. 
For simple questions, the model can often answer directly with sufficient accuracy, so calling a tool would be unnecessary and inefficient.


9. What are risks of tool misuse? 
Tool misuse can lead to several problems. 
The model might choose the wrong tool for a task or provide invalid inputs, resulting in incorrect outputs. 
It may also hallucinate arguments or rely too heavily on tool outputs without verifying them. 
In some cases, improper validation of tool inputs can create security risks, and careless use of tools could expose sensitive information.


10. How does prompt design affect decisions? 
Prompt design plays a critical role in how the agent behaves. Clear and specific instructions help the model understand when to use tools, which tool to select, and when to stop reasoning. 
A well-designed prompt improves consistency and reliability, while a vague or poorly structured prompt can lead to incorrect decisions, skipped tool usage, or unpredictable behavior.


11. What are limitations of this agent?
This agent has several limitations. 
Its performance depends heavily on the quality of retrieved information from the knowledge base, and it is restricted to the tools that are available. 
It may choose tools incorrectly or fail to complete reasoning within the allowed number of steps. 
Additionally, it cannot independently verify whether tool outputs are correct, and its knowledge is limited to what has been ingested into the system.