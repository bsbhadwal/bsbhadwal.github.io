---
layout: post
title: "The Scaling Dilemma: An Analysis of Tool Quantity and its Impact on ReAct Agent Performance"
---

## **Section 1: The ReAct Architecture and the Challenge of Scalable Tool Use**

### **1.1 Anatomy of a ReAct Agent: Deconstructing the Thought-Action-Observation Loop**

The ReAct (Reasoning and Acting) framework represents a foundational paradigm in the development of autonomous agents, enabling Large Language Models (LLMs) to solve complex, multi-step problems that exceed the capabilities of single-pass inference.1 At its core, ReAct introduces an iterative loop that emulates a simplified human problem-solving process. This cycle consists of three distinct phases:

**Thought, Action, and Observation**.

1. **Thought:** The agent first generates a reasoning trace. This internal monologue serves to decompose the user's query, assess the current state of the problem, formulate a plan, and decide on the next logical step.1  
2. **Action:** Based on its thought process, the agent selects and executes a specific action. In the context of modern agentic systems, this action is typically the invocation of an external tool, such as a search engine, calculator, or a custom API.4  
3. **Observation:** The execution of the action yields a result, or an "observation," which is fed back to the agent. This observation could be the output of an API call, data from a database, or an error message indicating that the action failed.2

This "Think-Act-Observe" sequence forms a powerful feedback loop. The observation from one step directly informs the thought process of the next, allowing the agent to dynamically adjust its plan, handle exceptions, and iteratively progress toward a final answer.4 Frameworks such as LangGraph are explicitly designed to manage these cyclical, stateful workflows. LangGraph models the agent as a graph where nodes represent functions (like generating a thought or executing a tool) and edges direct the flow of information. A central, shared "State" object persists context across multiple turns of the loop, ensuring the agent maintains a coherent understanding of the task as it unfolds.

### **1.2 The Promise and Peril of Large Toolsets: Introducing the "Paradox of Choice" for LLM Agents**

The theoretical promise of a ReAct agent is that its capabilities scale with the number of tools it can access. An agent equipped with a vast library of APIs and functions should, in principle, be able to solve a wider and more complex range of problems. However, empirical evidence reveals a significant and counterintuitive limitation: as the number of available tools increases, agent performance does not improve—it collapses.5

This phenomenon can be described as a "Paradox of Choice" for LLM agents.8 Faced with an overwhelming number of options, the LLM's reasoning engine becomes confused and less effective. Studies have shown that even with context windows large enough to accommodate extensive tool definitions, LLMs struggle to differentiate between similar tools, often select the wrong one, or fail to act entirely.8 This performance degradation is not a gradual decline but a sharp cliff. One study that scaled an agent's toolset to over 11,000 options found that its accuracy in selecting the correct tool fell to just 13.6%.5 Anecdotal evidence from practitioners further suggests that the practical limit for most models to reliably handle tools presented directly in-prompt is remarkably low, often hovering around just 12 to 16 distinct tools.5 This limitation presents a fundamental challenge to building versatile, enterprise-grade agents that must interact with a wide array of systems and APIs.

### **1.3 Defining Performance: A Multi-faceted View of Accuracy and Latency in Agentic Systems**

Evaluating the performance of a ReAct agent requires a more sophisticated framework than simply judging the correctness of its final answer. A truly robust and production-ready agent must be assessed across multiple dimensions of both accuracy and latency, as a correct final output can mask a brittle, inefficient, and costly underlying process.11 An agent that stumbles upon the correct solution after multiple failed attempts is not reliable. This understanding necessitates a shift in evaluation, where the quality of the process is as important as the quality of the result.

#### Accuracy Metrics

A comprehensive view of agent accuracy includes several layers of evaluation:

* **Final Answer Correctness:** This is the ultimate measure of task success. It assesses whether the agent's final response fully and accurately addresses the user's initial query. Given the generative nature of LLMs, this is often evaluated using a more powerful LLM-as-a-judge against a predefined rubric.7  
* **Tool Selection Accuracy:** This is a deterministic metric that checks if the agent selected the correct tool at each step of its reasoning process. It provides a direct measure of the LLM's ability to map its internal reasoning to the available external capabilities.13  
* **Trajectory Correctness:** This is the strictest form of evaluation, comparing the agent's sequence of tool calls against an optimal, ground-truth trajectory. It penalizes not only incorrect tool choices but also incorrect ordering, ensuring the agent is following the most logical path to a solution.7  
* **Tool-Calling Efficiency:** This metric quantifies the economy of the agent's actions. It can be measured by tracking redundant tool usage or by calculating the ratio of steps taken to the minimum expected steps. This distinguishes an agent that solves a problem efficiently from one that arrives at the same answer through a circuitous and wasteful path.13

#### Latency Metrics

Latency is a critical factor in user experience and operational cost. Key metrics include:

* **Time to First Token (TTFT):** This measures the delay between the user's request and the agent generating its first piece of output (e.g., the first word of its "Thought"). It is a primary indicator of the system's perceived responsiveness.16  
* **Time Per Output Token (TPOT) / Inter-Token Latency (ITL):** This measures the speed at which the agent generates subsequent tokens. A slow TPOT can make the agent's reasoning process appear sluggish to the end-user.16  
* **Total Task Completion Time:** This is the end-to-end latency, from the initial prompt to the delivery of the final answer. This metric is heavily influenced by the number of turns in the ReAct loop and the efficiency of each tool call, making it a crucial measure of overall system performance.15

The selection of which metrics to prioritize is not merely an academic exercise; it directly shapes architectural decisions. An organization that focuses solely on final-answer correctness might be led to develop a single, monolithic agent with an ever-expanding toolset. However, once metrics for latency, cost, and trajectory efficiency are introduced, the performance degradation associated with large toolsets becomes immediately apparent. This data-driven realization forces a re-evaluation of the monolithic approach and naturally guides architects toward more sophisticated, scalable patterns such as the multi-agent and retrieval-augmented systems discussed later in this report.

## **Section 2: Empirical Analysis of Tool Count on Agent Performance**

The inverse relationship between the number of available tools and a ReAct agent's performance is not merely theoretical. Multiple independent benchmarks provide quantitative evidence of this degradation across different models, tasks, and scales. These studies consistently show that as the toolset grows, accuracy declines while latency and cost increase, revealing a fundamental scaling challenge in agentic AI.

The following table consolidates the primary findings from three key studies, offering a direct, at-a-glance comparison of the impact of scaling toolsets on agent performance.

| Benchmark / Study | Model(s) Tested | Max Tools Tested | Key Accuracy Finding (Quantitative) | Key Latency/Efficiency Finding (Quantitative) | Source(s) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| LangChain ReAct Benchmark | claude-3.5-sonnet, gpt-4o, o1, o3-mini | 7 Domains (\~50+ tools) | Pass rate for gpt-4o dropped to 2% with 7 domains. o3-mini dropped from 68% to \<10%. | Longer trajectories (≥3 tool calls) degraded more quickly than shorter ones. | 7 |
| "Less is More" (Edge Devices) | Llama3.1-8b, Hermes2-Pro-8b, etc. | 46 tools | Success rate for Llama3.1-8b improved from \~20% to 44.2% when tools were dynamically reduced. | Execution time reduced by up to 70%; power consumption reduced by up to 40%. | 10 |
| RAG-MCP Study | GPT-4 (assumed) | 11,100 tools | Accuracy collapsed to 13.6% with all tools in-prompt; jumped to 43.1% with RAG-based retrieval. | Token usage was reduced from \>2,100 to \~1,080 on average. | 5 |

### **2.1 The Impact of Context Bloat on Latency and Cost**

The most immediate consequence of a large toolset is "context bloat." Each tool provided to an agent requires a definition, including its name, a natural language description of its purpose, and a structured schema for its parameters.19 All of this information consumes tokens within the LLM's limited context window.

This has a direct and measurable impact on latency. The initial processing of the input prompt, known as the "prefill" phase, is computationally intensive. A larger prompt, bloated with dozens or hundreds of tool definitions, significantly increases the time required for this phase, leading to a higher Time to First Token (TTFT).16 Subsequently, the model must reason over this expanded context during each step of the generation process, which slows down the rate at which it produces output tokens, thereby increasing the Time Per Output Token (TPOT).21 The cumulative effect is a substantial increase in the total time required to complete a task.

Furthermore, this inflation of token count has direct economic consequences. Most LLM providers bill based on the number of input and output tokens processed. A bloated prompt not only increases the input token count for the initial request but also tends to lengthen the reasoning traces in the model's output, further driving up costs for each agentic task.23

### **2.2 The Accuracy Cliff: Degradation in Tool Selection and Trajectory Correctness**

The data presented in the summary table illustrates that the decline in accuracy is not linear but often resembles a steep cliff. The LangChain benchmark provides a clear example: when tested on calendar scheduling tasks, gpt-4o's pass rate plummeted to just 2% when the number of irrelevant "distractor" domains was increased to seven. Similarly, the o3-mini model saw its performance drop from a respectable 68% to below 10% under the same conditions.7

This degradation is particularly pronounced for tasks that require more complex, multi-step reasoning. The same benchmark found that tasks requiring longer tool-calling trajectories (three or more tool calls) degraded at a much faster rate than simpler, shorter-trajectory tasks.7 This suggests a compounding error effect, where the increased probability of making a mistake at any single step dramatically lowers the likelihood of successfully completing the entire sequence.

These findings reveal a critical trade-off between an agent's potential capabilities and its practical reliability. Every tool added to an agent's context may introduce a new function, but it also imposes a "performance tax" on the entire system by making every existing function less reliable. This negative externality is a central challenge in scaling agentic systems. An engineer focused on adding a new tool for a specific business function (e.g., human resources) may not realize they are simultaneously degrading the performance of an entirely separate function (e.g., calendar scheduling). This hidden interdependency necessitates a holistic, system-level approach to agent design, where the decision to add a tool is weighed not only by its individual utility but also by its marginal impact on the reliability of the entire agent.

It is also important to note that this performance degradation is not uniform across all models. The LangChain benchmark showed that while all models suffered from context bloat, models like o1 and claude-3.5-sonnet demonstrated greater stability and resilience compared to gpt-4o and llama-3.3-70B, whose performance dropped off much more sharply.7 This suggests that robustness to "distractor" information within a large context is a key, differentiating capability among LLMs. The model that excels at simple, single-turn tasks may not be the optimal choice for complex, multi-tool agentic workflows, making model selection a critical architectural decision.

## **Section 3: Causal Mechanisms of Performance Degradation**

The observed decline in agent performance with an increasing number of tools is not an arbitrary phenomenon but is rooted in the fundamental architecture of LLMs and the cognitive challenges of complex decision-making. Understanding these causal mechanisms is essential for designing effective mitigation strategies.

### **3.1 The "Lost-in-the-Middle" Phenomenon: How Positional Bias Affects Tool Recall**

A primary architectural limitation of the Transformer models that power modern LLMs is a phenomenon known as positional bias, or the "lost-in-the-middle" problem. Research has demonstrated that these models tend to give disproportionate weight to information presented at the very beginning and very end of their context window, while often failing to recall or effectively utilize information located in the middle.15

This has direct implications for tool-augmented agents. When an agent is initialized with a long list of tool definitions, those tools that happen to fall in the middle of the prompt are at a significant disadvantage. The model is less likely to "see" them during its reasoning process, and therefore less likely to select them, even when they are the most appropriate tools for the given task.15 This provides a clear, architectural explanation for why simply expanding the context window with more tools does not lead to better performance; the model's ability to access information within that window is not uniform.

### **3.2 Cognitive Overload: The LLM's Diminishing Ability to Differentiate Between Tools**

Beyond the structural issue of positional bias, there is a semantic challenge that can be described as cognitive overload. As the number of tools increases, the semantic space they occupy becomes more crowded. The agent must differentiate between a larger set of options, many of which may have similar but subtly different functions or descriptions.8

This increased complexity can lead to two primary failure modes. First, the LLM may select a suboptimal but "close enough" tool because it fails to grasp the fine-grained distinctions between the best option and several other similar ones. Second, faced with an overwhelming number of choices, the model can become "paralyzed," failing to select any tool at all and halting the task.8 This issue is exacerbated when tool descriptions are not carefully engineered to be distinct, or when multiple tools offer overlapping capabilities, a common scenario in enterprise environments with legacy APIs.24 The quality and distinctiveness of tool descriptions are therefore critical variables; a large number of well-defined, semantically unique tools is less problematic than a smaller number of tools with ambiguous or overlapping descriptions.

### **3.3 Compounding Errors in Multi-Step Trajectories**

The performance degradation is most acute in complex tasks that require a sequence of multiple, dependent tool calls. This can be understood through a simple probabilistic model. If the probability of an agent selecting the correct tool at any given step is p, then the probability of successfully completing a task that requires n sequential steps is pn.

Even a small decrease in the single-step accuracy p—caused by the cognitive overload and positional bias from a large toolset—will lead to an exponential decay in the overall task success rate as the number of steps n increases. This mathematical reality explains the empirical finding from the LangChain benchmark that agents performing tasks with longer trajectories degraded more rapidly than those with shorter ones.7 This compounding error rate is a significant barrier to building agents that can reliably execute complex, multi-step plans.

This can create a vicious cycle within the agent's operation. When a tool selection fails, the agent may enter a retry loop, making another LLM call to "re-think" its approach.15 However, if the root cause of the initial failure was context overload, the subsequent reasoning steps are polluted by the same noisy context, making another failure likely. This leads to a negative feedback loop where the initial problem (too many tools) triggers behaviors (retries) that amplify the costs in terms of latency and token consumption, often without increasing the probability of success.21

## **Section 4: Advanced Architectural Patterns for High-Performance Agents**

To overcome the limitations of the single-agent, in-prompt tool paradigm, several advanced architectural patterns have emerged. These approaches move away from a "more is better" philosophy and instead focus on intelligently managing and presenting a smaller, more relevant set of tools to the LLM at any given time.

### **4.1 Dynamic Tool Retrieval: A Deep Dive into RAG-based Architectures**

The most direct solution to the problem of prompt bloat is to avoid placing the entire toolset into the prompt. Instead, a Retrieval-Augmented Generation (RAG) architecture can be employed to dynamically fetch only the most relevant tools for a given task.5

#### **4.1.1 Vector Search for Semantic Tool Selection**

The mechanics of a RAG-based tool retrieval system are as follows:

1. **Offline Indexing:** The descriptions and schemas of all available tools are processed by an embedding model and stored as high-dimensional vectors in a specialized vector database.27  
2. **Runtime Retrieval:** When a user query is received, it is also converted into a vector embedding using the same model.  
3. **Similarity Search:** A similarity search (e.g., cosine similarity or Euclidean distance) is performed in the vector database to find the tool vectors that are semantically closest to the query vector.30  
4. **Prompt Injection:** The descriptions of the top-k most relevant tools are retrieved and injected into the agent's prompt. The agent then performs its reasoning and tool selection process on this much smaller, highly relevant subset of tools.31

This approach was validated in the RAG-MCP study, which found that retrieving relevant tools for a pool of 11,100 options increased tool selection accuracy from 13.6% to 43.1% while simultaneously cutting average token usage in half.5

#### **4.1.2 Performance Trade-offs of Retrieval**

This architecture introduces a new source of latency: the retrieval step from the vector database. However, modern vector databases are highly optimized for this task, with performance typically measured in milliseconds. Key metrics for evaluating these systems include P95/P99 latency (the time within which 95% or 99% of queries complete) and Queries Per Second (QPS) under concurrent load.27 The critical insight is that the small, predictable latency of a vector search is a highly favorable trade-off compared to the significant and often unpredictable latency incurred by forcing an LLM to reason over a massively bloated and distracting context window.26

### **4.2 Hierarchical and Multi-Agent Frameworks**

An alternative "divide and conquer" strategy involves decomposing a complex problem space into smaller, more manageable domains, each handled by a specialized agent. Instead of a single, monolithic agent responsible for all tasks, a system of collaborating agents can be orchestrated to achieve the goal.35

#### **4.2.1 The Supervisor-Worker Model**

A common implementation of this pattern is the supervisor-worker model. In this architecture, a high-level "supervisor" or "planner" agent receives the initial user request. Its sole responsibility is to analyze the request, decompose it into sub-tasks, and route each sub-task to an appropriate "worker" agent.37 Each worker agent is equipped with a small, curated set of tools relevant only to its specific domain (e.g., a "finance agent" with financial APIs, a "customer support agent" with CRM tools).

This approach was explicitly tested in a LangChain benchmark, which demonstrated that while a single agent's performance "falls off sharply" when distractor domains are added, the multi-agent supervisor system remains robust and maintains a higher level of performance.37 This pattern is already being applied in real-world enterprise scenarios; for example, Fujitsu developed a system for generating sales proposals that uses specialized agents for distinct sub-tasks like data analysis, market research, and document creation.35

### **4.3 Hybrid and Two-Stage Selection Models**

A more nuanced approach combines elements of routing and retrieval. This pattern uses a two-stage process for tool selection. First, a smaller, faster, and more cost-effective LLM acts as a "tool-selector" or "router".21 Its job is not to execute the task but to perform a rapid, coarse-grained analysis of the user's query and identify a smaller, relevant subset of tools or a specific capability that is needed.41

This pre-filtered list of tools, or a prompt to invoke the identified capability, is then passed to a larger, more powerful LLM. The primary LLM can then perform its detailed, fine-grained reasoning on a much cleaner and more focused context. This architecture effectively decouples the broad task of "capability identification" from the more detailed task of "tool implementation and execution," optimizing the entire workflow for both speed (by using a fast model for the initial triage) and accuracy (by using a powerful model for the core reasoning).24

The optimal architectural choice is contingent on the structure of the tool space itself. For toolsets that are highly diverse and span multiple, distinct domains (e.g., calendar, HR, finance), a hierarchical multi-agent approach is often superior. The task of routing a query to the correct domain expert is a well-defined classification problem. Conversely, for toolsets that are large but semantically concentrated within a single domain (e.g., hundreds of different ways to query a single complex API), a RAG-based retrieval approach is more suitable, as it can discern the subtle semantic differences between similar tools.

These advanced architectures are not mutually exclusive and can be composed into highly effective hybrid systems. A state-of-the-art, large-scale agentic system could employ a two-stage router to first select a specialized sub-agent from a roster of experts. This top-level routing decision is a low-cardinality problem, as the router only needs to choose from a small number of agents. Once activated, that specialist agent could then use its own dedicated RAG system to retrieve the most relevant tools from its curated, domain-specific tool library. This multi-level process of filtering and retrieval dramatically simplifies the decision-making at each stage, creating a system that is more scalable, robust, and efficient.

## **Section 5: Strategic Recommendations for Designing and Deploying Robust ReAct Agents**

The transition from simple, proof-of-concept agents to robust, production-grade systems requires a deliberate architectural strategy that directly confronts the "Paradox of Choice." The following recommendations provide a framework for managing toolsets and designing agents that are both capable and reliable.

### **5.1 A Decision Framework for Toolset Management**

The optimal architecture for a ReAct agent is not one-size-fits-all; it depends on the scale and nature of the tools it must manage. The following criteria can guide the selection of an appropriate design pattern:

* **Number of Tools:** For a very small number of tools (e.g., fewer than 15), a simple, single-agent architecture with all tools defined in the prompt may be sufficient and is the easiest to implement. Once the toolset grows beyond this approximate threshold, performance is likely to degrade, necessitating a move to more advanced patterns.  
* **Tool Diversity (Inter-Domain Complexity):** If the available tools span multiple, distinct functional domains (e.g., scheduling, customer support, and financial analysis), a **hierarchical or multi-agent architecture** is the most effective approach. This allows a supervisor agent to route tasks to specialized worker agents, each with a small, focused toolset, thereby isolating and simplifying the context for each decision.  
* **Tool Similarity (Intra-Domain Complexity):** If the toolset consists of a large number of tools within a single, semantically-related domain (e.g., 100 different API endpoints for a single enterprise application), a **RAG-based dynamic tool retrieval** architecture is preferable. This approach uses semantic search to identify the subtle differences between similar tools and provides only the most relevant options to the agent.  
* **Latency Requirements:** For applications requiring real-time interaction and minimal response times, a **two-stage selection model** offers significant advantages. Using a smaller, faster model for the initial routing or retrieval step can dramatically reduce the Time to First Token (TTFT), improving the user's perception of the system's responsiveness.

### **5.2 Best Practices for Tool and Prompt Engineering**

The performance of any agent architecture is heavily dependent on the quality of the inputs provided to the LLM.

* **Craft High-Quality Tool Descriptions:** Tool descriptions are the primary interface between the LLM's reasoning process and its capabilities. Descriptions must be clear, concise, and, most importantly, semantically distinct. They should accurately convey the tool's purpose, its expected inputs, and its outputs. The quality of these descriptions directly impacts the performance of both the LLM's internal tool selection logic and the accuracy of any external RAG retrieval system.  
* **Structure Prompts to Mitigate Bias:** To counteract the "lost-in-the-middle" problem, structure prompts strategically. Place the most critical system-level instructions and a few illustrative examples (few-shot prompting) at the very beginning and end of the prompt, as this is where the model's attention is highest.  
* **Curate, Don't Accumulate:** Resist the temptation to simply add every available API endpoint as a tool. A "more is better" approach is a proven anti-pattern.5 Instead, actively curate the toolset. Regularly review tool usage and remove redundant or infrequently used tools. Consider creating higher-level, composite tools that abstract away sequences of low-level API calls. This practice of "tool curation" reduces the cognitive load on the agent, improving the reliability of the entire system.

### **5.3 Future Outlook: Towards More Intelligent and Efficient Agents**

The challenges outlined in this report are an active area of research. One promising avenue is the fine-tuning of smaller, specialized models specifically for the task of tool selection.10 A highly optimized, fine-tuned model could potentially outperform a much larger, general-purpose model at the specific task of routing or tool retrieval, offering a path to even lower latency and higher accuracy.

In conclusion, as agentic systems are deployed to solve increasingly complex, real-world problems, the limitations of the naive single-agent, in-prompt tool architecture become starkly apparent. It is insufficient for building reliable, scalable, and efficient production-grade systems. The future of effective agentic AI lies in more sophisticated architectures that actively manage the "Paradox of Choice." By implementing dynamic retrieval, hierarchical delegation, and intelligent routing, developers can build the next generation of AI agents that are not only powerful in theory but also robust and performant in practice.

#### **Works cited**

1. ReAct \- Prompt Engineering Guide, accessed on August 15, 2025, [https://www.promptingguide.ai/techniques/react](https://www.promptingguide.ai/techniques/react)  
2. LLM Agents → ReAct, Toolformer, AutoGPT family & Autonomous Agent Frameworks | by Akanksha Sinha | Medium, accessed on August 15, 2025, [https://medium.com/@akankshasinha247/react-toolformer-autogpt-family-autonomous-agent-frameworks-2c4f780654b8](https://medium.com/@akankshasinha247/react-toolformer-autogpt-family-autonomous-agent-frameworks-2c4f780654b8)  
3. What Are AI Agents? | IBM, accessed on August 15, 2025, [https://www.ibm.com/think/topics/ai-agents](https://www.ibm.com/think/topics/ai-agents)  
4. ReAct agents vs function calling agents \- LeewayHertz, accessed on August 15, 2025, [https://www.leewayhertz.com/react-agents-vs-function-calling-agents/](https://www.leewayhertz.com/react-agents-vs-function-calling-agents/)  
5. Too Many Tools Break Your LLM : r/mcp \- Reddit, accessed on August 15, 2025, [https://www.reddit.com/r/mcp/comments/1m9227n/too\_many\_tools\_break\_your\_llm/](https://www.reddit.com/r/mcp/comments/1m9227n/too_many_tools_break_your_llm/)  
6. Too Many Tools? How LLMs Struggle at Scale | MCP Talk w/ Matthew Lenhard \- YouTube, accessed on August 15, 2025, [https://www.youtube.com/watch?v=ej7-n9OoGnQ](https://www.youtube.com/watch?v=ej7-n9OoGnQ)  
7. Benchmarking Single Agent Performance \- LangChain Blog, accessed on August 15, 2025, [https://blog.langchain.com/react-agent-benchmarking/](https://blog.langchain.com/react-agent-benchmarking/)  
8. Why Can't I Just Use an API? Because Your AI Agent Needs MCP \- Auth0, accessed on August 15, 2025, [https://auth0.com/blog/mcp-vs-api/](https://auth0.com/blog/mcp-vs-api/)  
9. Paradox of Choice in the AI Universe | by Prasanna Vee \- Medium, accessed on August 15, 2025, [https://prasannavee.medium.com/paradox-of-choice-in-the-universe-of-ai-19fe25417897](https://prasannavee.medium.com/paradox-of-choice-in-the-universe-of-ai-19fe25417897)  
10. Less is More: Optimizing Function Calling for LLM Execution on Edge Devices \- arXiv, accessed on August 15, 2025, [https://arxiv.org/html/2411.15399v1](https://arxiv.org/html/2411.15399v1)  
11. 7 Key LLM Metrics to Enhance AI Reliability | Galileo, accessed on August 15, 2025, [https://galileo.ai/blog/llm-performance-metrics](https://galileo.ai/blog/llm-performance-metrics)  
12. LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide \- Confident AI, accessed on August 15, 2025, [https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)  
13. LLM Agent Evaluation: Assessing Tool Use, Task Completion, Agentic Reasoning, and More, accessed on August 15, 2025, [https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)  
14. Benchmarking Agent Tool Use \- LangChain Blog, accessed on August 15, 2025, [https://blog.langchain.com/benchmarking-agent-tool-use/](https://blog.langchain.com/benchmarking-agent-tool-use/)  
15. Evaluating Agent Tool Selection — Testing if First Really is the Worst | by ODSC, accessed on August 15, 2025, [https://odsc.medium.com/evaluating-agent-tool-selection-testing-if-first-really-is-the-worst-b83dad43f641](https://odsc.medium.com/evaluating-agent-tool-selection-testing-if-first-really-is-the-worst-b83dad43f641)  
16. Best Practices for LLM Latency Benchmarking | newline \- Fullstack.io, accessed on August 15, 2025, [https://www.newline.co/@zaoyang/best-practices-for-llm-latency-benchmarking--257f132d](https://www.newline.co/@zaoyang/best-practices-for-llm-latency-benchmarking--257f132d)  
17. Benchmarking LLM Serving Performance: A Comprehensive Guide | by Doil Kim \- Medium, accessed on August 15, 2025, [https://medium.com/@kimdoil1211/benchmarking-llm-serving-performance-a-comprehensive-guide-db94b1bfe8cf](https://medium.com/@kimdoil1211/benchmarking-llm-serving-performance-a-comprehensive-guide-db94b1bfe8cf)  
18. Less is More: Optimizing Function Calling for LLM Execution on ..., accessed on August 15, 2025, [https://arxiv.org/pdf/2411.15399](https://arxiv.org/pdf/2411.15399)  
19. Calculating LLM Token Counts: A Practical Guide \- Winder.AI, accessed on August 15, 2025, [https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/](https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/)  
20. What Is Tool Calling? | IBM, accessed on August 15, 2025, [https://www.ibm.com/think/topics/tool-calling](https://www.ibm.com/think/topics/tool-calling)  
21. Latency optimization \- OpenAI API, accessed on August 15, 2025, [https://platform.openai.com/docs/guides/latency-optimization](https://platform.openai.com/docs/guides/latency-optimization)  
22. LLM Latency Benchmark by Use Cases in 2025 \- Research AIMultiple, accessed on August 15, 2025, [https://research.aimultiple.com/llm-latency-benchmark/](https://research.aimultiple.com/llm-latency-benchmark/)  
23. The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective \- arXiv, accessed on August 15, 2025, [https://arxiv.org/html/2506.04301v1](https://arxiv.org/html/2506.04301v1)  
24. Tool Selection by Large Language Model (LLM) Agents \- Technical Disclosure Commons, accessed on August 15, 2025, [https://www.tdcommons.org/cgi/viewcontent.cgi?article=9446\&context=dpubs\_series](https://www.tdcommons.org/cgi/viewcontent.cgi?article=9446&context=dpubs_series)  
25. MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch \- arXiv, accessed on August 15, 2025, [https://arxiv.org/html/2506.01056v1](https://arxiv.org/html/2506.01056v1)  
26. Automatic Tool Selection to Reduce Large Language Model Latency \- Technical Disclosure Commons, accessed on August 15, 2025, [https://www.tdcommons.org/cgi/viewcontent.cgi?article=8702\&context=dpubs\_series](https://www.tdcommons.org/cgi/viewcontent.cgi?article=8702&context=dpubs_series)  
27. Announcing VDBBench 1.0: Open-Source Vector Database Benchmarking with Your Real-World Production Workloads \- Milvus, accessed on August 15, 2025, [https://milvus.io/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads.md](https://milvus.io/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads.md)  
28. Enhancing LLM Performance with Retrieval-Augmented Generation \- RSNA, accessed on August 15, 2025, [https://www.rsna.org/news/2025/august/retrieval-augmented-generation](https://www.rsna.org/news/2025/august/retrieval-augmented-generation)  
29. LLM vector database: Why it's not enough for RAG, accessed on August 15, 2025, [https://www.k2view.com/blog/llm-vector-database/](https://www.k2view.com/blog/llm-vector-database/)  
30. Understanding Vector Databases: The Secret to Advanced AI Searches \- Aerospike, accessed on August 15, 2025, [https://aerospike.com/blog/vector-database-ai-search/](https://aerospike.com/blog/vector-database-ai-search/)  
31. BigTool: Agents with large number of tools \- YouTube, accessed on August 15, 2025, [https://www.youtube.com/watch?v=3ISRS2hQlfI](https://www.youtube.com/watch?v=3ISRS2hQlfI)  
32. ANN Benchmark | Weaviate Documentation, accessed on August 15, 2025, [https://docs.weaviate.io/weaviate/benchmarks/ann](https://docs.weaviate.io/weaviate/benchmarks/ann)  
33. Vector Database Benchmarks: A Definitive Guide to Tools, Metrics, and Top Performers, accessed on August 15, 2025, [https://medium.com/@vkmauryavk/vector-database-benchmarks-a-definitive-guide-to-tools-metrics-and-top-performers-4c4110e61f73](https://medium.com/@vkmauryavk/vector-database-benchmarks-a-definitive-guide-to-tools-metrics-and-top-performers-4c4110e61f73)  
34. Local LLM Tool Calling: Which LLM Should You Use? | DockerTool Calling with Local LLMs: A Practical Evaluation | Docker, accessed on August 15, 2025, [https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)  
35. Agent Factory: The new era of agentic AI—common use cases and design patterns, accessed on August 15, 2025, [https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/](https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/)  
36. Benchmarking Multi-Agent AI: Insights & Practical Use | Galileo, accessed on August 15, 2025, [https://galileo.ai/blog/benchmarks-multi-agent-ai](https://galileo.ai/blog/benchmarks-multi-agent-ai)  
37. Benchmarking Multi-Agent Architectures \- LangChain Blog, accessed on August 15, 2025, [https://blog.langchain.com/benchmarking-multi-agent-architectures/](https://blog.langchain.com/benchmarking-multi-agent-architectures/)  
38. LLM agents: The ultimate guide 2025 | SuperAnnotate, accessed on August 15, 2025, [https://www.superannotate.com/blog/llm-agents](https://www.superannotate.com/blog/llm-agents)  
39. Agent-as-Tool: A Study on the Hierarchical Decision Making with Reinforcement Learning, accessed on August 15, 2025, [https://arxiv.org/html/2507.01489v1](https://arxiv.org/html/2507.01489v1)  
40. Agent-as-Tool: A Study on the Hierarchical Decision Making with Reinforcement Learning, accessed on August 15, 2025, [https://www.researchgate.net/publication/393333947\_Agent-as-Tool\_A\_Study\_on\_the\_Hierarchical\_Decision\_Making\_with\_Reinforcement\_Learning](https://www.researchgate.net/publication/393333947_Agent-as-Tool_A_Study_on_the_Hierarchical_Decision_Making_with_Reinforcement_Learning)  
41. Multi-LLM routing strategies for generative AI applications on AWS | Artificial Intelligence, accessed on August 15, 2025, [https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/)  
42. Doing More with Less – Implementing Routing Strategies in Large Language Model-Based Systems: An Extended Survey \- arXiv, accessed on August 15, 2025, [https://arxiv.org/html/2502.00409v2](https://arxiv.org/html/2502.00409v2)  
43. What is the maximum number of tools that can be added to an agent : r/AI\_Agents \- Reddit, accessed on August 15, 2025, [https://www.reddit.com/r/AI\_Agents/comments/1lwxh4r/what\_is\_the\_maximum\_number\_of\_tools\_that\_can\_be/](https://www.reddit.com/r/AI_Agents/comments/1lwxh4r/what_is_the_maximum_number_of_tools_that_can_be/)  
44. GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation \- arXiv, accessed on August 15, 2025, [https://arxiv.org/html/2502.18990v1](https://arxiv.org/html/2502.18990v1)
