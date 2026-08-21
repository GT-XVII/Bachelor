Title of the thesis	
Comparison of a Classical RAG Pipeline and an Agent-Based Approach for Intelligent Knowledge Retrieval Across Enterprise Documentation Sources

Motivation, problem statement, expected result	
In many enterprise environments, relevant knowledge is distributed across multiple systems such as Jira, Confluence, SharePoint, internal portals, and local documents. This fragmentation makes efficient information retrieval difficult, especially when team members need fast, reliable, and context-aware answers to operational or client-related questions. Manual search across these heterogeneous sources is time-consuming, error-prone, and often depends heavily on individual experience.

A previous project addressed this challenge by extending the Ask Your Template (AYT) platform, a Retrieval-Augmented Generation (RAG) pipeline, with Jira and Confluence integration. In this setup, knowledge objects are ingested, normalized, chunked, vectorized, and made available for semantic retrieval. While this approach proved technically feasible and useful, it also revealed limitations regarding chunking strategy, source-specific retrieval quality, setup complexity, and long-term maintainability.

At the same time, agent-based AI systems have recently gained relevance as an alternative paradigm for intelligent information access. Instead of relying only on retrieval over vectorized chunks, such systems can dynamically decide which source to query, how to decompose tasks, and how to combine retrieved information.

Research Questions
This thesis addresses the following research questions:
•	How does a classical RAG pipeline compare to an agent-based approach in terms of latency, cost, and retrieval performance in enterprise environments?
•	To what extent does an agent-based system improve source selection and answer quality compared to RAG?
•	What are the differences in implementation complexity and maintainability between both approaches?
•	Under which conditions is each approach preferable in practice?

The objective of this thesis is to design and implement a comparative evaluation of both approaches using a shared enterprise use case. The expected result is a structured analysis identifying strengths, weaknesses, and trade-offs, providing a decision basis for selecting an appropriate architecture for intelligent knowledge systems in environments such as the TAP Rollout Team.

Planned procedure	
The thesis follows a comparative experimental research approach with prototype-based implementation, combining elements of design science and empirical evaluation.

First, the theoretical foundations of enterprise knowledge retrieval, Retrieval-Augmented Generation, and agent-based systems are established through a structured literature review. Based on this, evaluation criteria such as latency, cost, retrieval quality, implementation complexity, and maintainability are formally defined.

Next, the existing RAG-based solution is analyzed as a reference architecture. In parallel, an agent-based system for the same retrieval task is designed and implemented. Both systems are modeled using structured system representations (e.g., architecture diagrams and data flow diagrams) to ensure comparability.

The evaluation is conducted using a predefined set of representative enterprise queries. The following methods are applied:
•	Comparative analysis of both systems under identical conditions
•	Experimental evaluation using measurable metrics:
o	Latency (response time per query)
o	Cost (API/token usage estimation)
o	Retrieval quality (defined query set with structured evaluation criteria)
•	Qualitative analysis of maintainability and implementation complexity

The results are systematically compared to derive a structured assessment of both approaches and identify their respective strengths, weaknesses, and suitable application scenarios.

Methodological Competencies and Application of Study Content

The thesis explicitly applies competencies acquired during the study program, particularly in software engineering, data processing, and system design.

The following methodological competencies are applied:
•	Process modeling and structured development approaches (e.g., phased implementation and evaluation workflow)
•	System modeling techniques, including architecture and data flow representations for both RAG and agent-based systems
•	Evaluation methodology, including definition of measurable criteria and structured experimental setup
•	Software engineering practices, such as modular system design and reproducible implementation

Additionally, concepts from machine learning and information retrieval, such as vector embeddings, semantic search, and agent-based decision logic, are applied in a practical context.

Literature list
(z.B. assigned literature, fundamental literature, literature pf preparation, advanced literature)	

Assigned literature / fundamental literature

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W. t., Rocktäschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems.

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., and Wang, H. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv.

Mialon, G., Dessì, R., Lomeli, M., Ebrahimi, M., and others. (2023). Augmented Language Models: a Survey. arXiv.

Russell, S., and Norvig, P. (2021). Artificial Intelligence: A Modern Approach. 4th ed. Pearson.

Literature for preparation

Atlassian. Jira Software Documentation. Official documentation.

Atlassian. Confluence Documentation. Official documentation.

Apache Lucene / vector search / embedding documentation depending on the chosen technical stack.

Documentation of the AYT platform and internal project materials from the previous Jira and Confluence integration work.

Advanced literature

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv.

Qin, Y., Zhang, S., Chen, D., and others. (2023). Tool Learning with Foundation Models. arXiv.

Xi, Z., Chen, W., Guo, X., and others. (2023). The Rise and Potential of Large Language Model Based Agents: A Survey. arXiv.

Chase, H. LangChain Documentation or equivalent framework documentation if used for the agent implementation.

LanceDB, FAISS, or chosen vector database documentation depending on implementation.

