arXiv:2601.07711v2 [cs.CL] 20 Apr 2026
Is Agentic RAG worth it? An experimental comparison of RAG approaches
Pietro Ferrazzi1,2
, Milica Cvjeticanin3
, Alessio Piraccini4
, Davide Giannuzzi5
1Fondazione Bruno Kessler, Trento, Italy
2University of Padova, Italy
3Cargill Geneve, Switzerland
4Alkemy, Milan, Italy
5Komebi Studio, Milan, Italy
Correspondence: pferrazzi [at] fbk [dot] eu
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems are usually defined by the combination
of a generator and a retrieval component that
extracts textual context from a knowledge base
to answer user queries. However, such ba-
sic implementations exhibit several limitations,
including noisy or suboptimal retrieval, mis-
use of retrieval for out-of-scope queries, weak
query–document matching, and variability or
cost associated with the generator. These short-
comings have motivated the development of
"Enhanced" RAG, where dedicated modules
are introduced to address specific weaknesses
in the workflow. More recently, the growing
self-reflective capabilities of Large Language
Models (LLMs) have enabled a new paradigm,
often referred to as "Agentic" RAG. In this
approach, an LLM orchestrates the entire pro-
cess, deciding which actions to perform, when
to perform them, and whether to iterate. De-
spite the rapid adoption of both paradigms, it re-
mains unclear which approach is preferable un-
der which conditions. In this work, we conduct
an empirically driven evaluation of "Enhanced"
and "Agentic" RAG across multiple scenarios
and dimensions. Our results provide practi-
cal insights into the trade-offs between the two
paradigms, offering guidance on selecting the
most effective RAG design for real-world ap-
plications, considering both performance and
costs.
applications like enterprise QA, search assistants,
internal knowledge bots (IBM, 2025; AWS, 2025;
Azure, 2025). Since the first initial definitions,
RAG workflows have been expanded to the so-
called Enhanced RAG (Figure 1, left). Such sys-
tems add to the retrieval and generation blocks com-
ponents that perform further refinement. Recently,
LLMs’ increasing self-reflective capabilities have
enabled a shift towards Agentic RAG (Figure 1,
right), where the LLM acts as an orchestrator, de-
ciding which actions to perform, utilizing different
tools for different purposes. Such systems are no
longer fixed pipelines, but rather iterative loops
guided by the model itself. Although initial work
on identifying theoretical distinctions between En-
hanced and Agentic RAG systems has been pro-
posed (Neha and Bhati, 2025), it remains unclear
what the performance differences are between the
two systems. To this end, we aim to extract action-
able insights for practitioners by analyzing perfor-
mances and costs. Our research question can be
stated as follow:
When designing RAG systems, should prac-
titioners adopt Agentic architectures or more
traditional (Enhanced) pipelines?
1 Introduction
Retrieval-Augmented Generation (RAG) has
evolved from a research concept (Lewis et al.,
2020) into a core component of production-grade
language systems, playing a central role in driv-
ing digital transformation across organizations (Ar-
slan et al., 2024). This shift has fostered atten-
tion by both the research community (Wang et al.,
2024; Fan et al., 2024) and industry, with cloud
providers offering their own RAG solutions for
Our first contribution consists of an experiment-
driven comparison of the two paradigms in four
dimensions relevant to production environments
(Table 1). Our second contribution consists of a
detailed analysis of costs and computational time
required by the two systems under several scenar-
ios. Finally, we propose a practical summary of our
findings, aiming to support informed architectural
choices in real-world RAG deployments1
.
1As per track requirements: preprint at https://arxiv.
org/abs/2601.07711, Industry Day at LREC2026
Figure 1: Left— Enhanced RAG. The system is composed by a sequence of modules, each responsible for
improving a specific stage of the RAG pipeline. A router determines whether a query should trigger retrieval; a
rewriter reformulates the query; a retriever selects candidate chunks from the knowledge base; and a reranker orders
the retrieved context before passing it to the generator. The workflow is fixed: information flows through predefined
blocks intended to mitigate known weaknesses of naïve RAG systems (defined by the simple composition of the
retriever and generator blocks). Right— Agentic RAG. The LLM acts as an agent that orchestrates the entire
process. At each step, it can choose to call a RAG tool or proceed to answer generation. Retrieval and context
refinement can be repeated, as the agent autonomously selects operations based on the evolving state of the task.
Naïve RAG shortcoming What we evaluate Enhanced Implementations
Agentic
Retrieval is performed even for
queries that do not require it
Accuracy of RAG usage for
in-scope and out-of-scope queries
Semantic routing
system
Agent decides whether
to do retrieval or not
Queries and documents in the KB
differ in format or semantics,
causing weak retrieval
Impact of query rewriting
techniques
Hyde-based query
rewriting
Agent rewrites query
as it wishes
Noisy or suboptimal retrieval Impact of retrieved document list
refinement techniques
Encoder-based
re-ranker
Agent can redo
retrieval multiple times
The underlying LLM is too weak
/ too slow / is too expensive
Impact of selecting more / less
powerful models Test different LLMs Test different LLMs
Table 1: Summary of the evaluation dimensions we select. For each shortcoming in Naïve RAG, we define
an evaluation dimension ("What we evaluate") and an implementation to test how Enhanced and Agentic RAG
overcome such a limitation.
2 Related work
RAG The concept of RAG, first introduced by
Lewis et al. (2020), has undergone intensive re-
search. Comprehensive overviews are presented
by Gao et al. (2023b); Fan et al. (2024); Wang
et al. (2024). As large language models (LLMs)
have acquired the capacity to multi-step reason-
ing and reflection, their consistency has enabled
a paradigm shift toward Agentic RAG solutions
(Shinn et al., 2023; Madaan et al., 2023). An
overview of how to combine the reasoning capabili-
ties of LLMs with RAG-like structures is presented
by Li et al. (2025b). While the definition of the
properties that characterize AI agents has evolved
(Masterman et al., 2024), this emerging research
direction has not yet been comprehensively cat-
egorized within a unified taxonomy, with initial
attempts made by Singh et al. (2025); de Aquino e
Aquino et al. (2025).
RAG in enterprise Industry reports consistently
identify knowledge-grounded applications—such
as question answering over proprietary data, en-
terprise search, and document understanding—as
among the highest-value use cases of generative AI.
Reports by McKinsey (2023), and Deloitte (2024)
emphasize the importance of connecting language
models to internal data sources to improve relia-
bility and business impact. Examples of such are
use cases where large databases of user-technician
interactions on specific issues are leveraged to pro-
vide answers to new users’ requests; applications to
query the all set of internal resources by companies
(Giulia Rutigliano, 2023). Notably, the workshop
on Generative AI and RAG Systems for Enterprise
at CIKM’24 (Xu et al., 2024) collects a series of
applications in enterprise settings that motivate our
focus on RAG itself.
Terms definitions Naïve Rag (Gao et al., 2023b)
is the simplest instantiation of the RAG paradigm,
where a retrieval step extracts a fixed number of
documents to be combined with the query and
passed to an LLM for the answer generation step.
According to the taxonomy outlined in Huang and
Huang (2024), Enhanced RAG refers to any Naïve
RAG pipeline augmented with additional steps
designed to improve its. Relevant examples are
CRAG (Yan et al., 2024), Self-RAG (Asai et al.,
2024), RaCoT (Cai et al., 2026), and INSIGHT-
RAG (Chen et al., 2026), which all focus on en-
hancing the performances of basic pipelines. Agen-
tic RAG (Yao et al., 2023; Li et al., 2025a; Alzubi
et al., 2025) is a system in which the LLM assumes
control over the workflow, being able to dynami-
cally decide to perform actions.
The need for empirical comparison A prelimi-
nary effort for experimental comparisons between
Agentic and Enhanced RAG is presented by Neha
and Bhati (2025), who propose a set of definitions
and evaluation dimensions but stop short of con-
ducting a full empirical study. Others (Xi et al.,
2025; Yang et al., 2024; Chen et al., 2024) limit the
benchmarking to one of the two settings.
Agents design and implementation Several
open-source frameworks have emerged to support
the development of Agentic RAG systems (Ta-
ble 8), reflecting the rapid growth of this area, rang-
ing from minimalist designs to rich abstractions.
3 Evaluation
The evaluation of RAG systems has been frequently
decomposed into assessments of individual sub-
components (Es et al., 2024; Chen et al., 2020),
each corresponding to a dimension that influences
overall effectiveness. We design our evaluation
following the same approach. First, we identify a
list of limitations of Naïve RAG pipelines based
on the work done by Huang and Huang (2024),
which we formulate in Table 1. Then, we con-
struct an experimental setting to compare how two
implementations of Enhanced and Agentic RAG
systems2 address each of them. We focus on single-
tool Agentic RAG systems, where the agent can
only decide to invoke retrieval or produce a final an-
swer. This design choice keeps its functional scope
comparable to Enhanced RAG pipelines: multi-
tool agents introduce additional capabilities (e.g.,
planning, external APIs) that would confound the
comparison. In the following sections, for each of
the four identified dimensions, we i) define it, ii)
detail our implementation choices, iii) present the
evaluation setting, iv) define the evaluation metrics.
3.1 Evaluation Datasets
To conduct our experiments, we require datasets
that are representative of common RAG applica-
tions, consisting of queries paired with a knowl-
edge base. Following the taxonomy proposed
by Arslan et al. (2024), which categorizes RAG
use cases by application area, we focus on the
most prominent natural language based scenarios.
Specifically, we consider the two major categories:
i) Question Answering (QA), where RAG is uti-
lized to ground answers in factual knowledge, and
ii) Information Retrieval and Extraction (IR/E),
where RAG is intended as a tool to get knowledge
from data through natural language queries.
For QA, we selected FIQA (Maia et al., 2018) and
NQ (Kwiatkowski et al., 2019) in the version re-
leased by Thakur et al. (2021). For IR/E, we used
FEVER (Thorne et al., 2018) and CQADupStack-
English (Hoogeveen et al., 2015). Each dataset is
chosen to represents a different real-world scenario
and task type, as described in Table 2.
3.2 User Intent Handling
Definition We refer to user intent handling as the
need of determining whether a certain query re-
quires the usage of retrieval or not. While prior sur-
veys on RAG (Huang and Huang, 2024; Gao et al.,
2023b; Fan et al., 2024) do not explicitly address
nor mention it, Wang et al. (2024) highlight its im-
portance by proposing a dedicated classifier for this
task. We argue that intent detection is crucial in
real-world RAG systems, as it prevents unneces-
sary or inappropriate retrieval calls. When a query
is classified as out-of-scope, the subsequent system
behavior is application-dependent (e.g., fallback
responses, refusal, or parametric answering), and
is therefore beyond the scope of this work, which
2We build Agentic Rag on PocketFlow (the-pocket, 2025)
Task Domain Dataset #Query #Doc Avg
D/Q Task description
QA General NQ 3,452 2,681,468 1.2 Broad QA use cases, where users can ask any type of question
to be answered via knowledge retrieval
QA Finance FiQA 648 57,638 2.6 Domain-specific queries to be answered by grounding re-
sponses on expert knowledge
IE/R Grammar
forum
CQAD-
EN 1,570 40,221 1.4 Find previously resolved blog posts that address the same
question posed by the user, providing user-friendly summary
IE/R Wikipedia FEVER 6,666 5,416,568 1.2
Seek evidence for or against the user statement ("claim veri-
fication") by finding documents and returning a final assess-
ment with a summary of the references
Table 2: Overview of the four selected datasets used for the experimental settings. For both Question Answering
(QA) and Information Extraction and Retrieval (IE/R), 2 datasets are selected. Each query has a labelled list of
relevant and irrelevant documents. Avg. D/Q indicates the average number of relevant documents per query.
focuses solely on the routing decision itself.
Enhanced Implementation We implement an
Enhanced RAG routing system using the semantic-
router framework (Aurelio Labs, 2025) A router is
defined by two sets of example queries, labelled as
valid and invalid respectively. At inference time,
the user query is compared to these groups and it is
classified as valid or invalid accordingly. The sys-
tem uses RAG to answer valid queries and avoids
it for invalid ones. For our experiments, we uti-
lize OpenAI’s text-embedding-3-small as em-
bedder. More details on the structure of the routing
system are reported in Appendix A.4.
Agentic Implementation An Agentic RAG sys-
tem embeds the ability to discriminate between
queries that require retrieval by design. When
a query is received, the agent can freely decide
whether to utilise the RAG node or answer directly.
Experimental setting We tested performances
on a dataset composed by an equal number of
valid and invalid queries (500 for each of the four
datasets). We selected the valid queries from the
train splits of each dataset, while we generated
the invalid ones prompting gpt-4o via 5-shot. We
validate the invalid queries generation by calcu-
lating their average similarity with the valid ones
(Appendix A.2). We make publicly available the
datasets of valid and invalid queries3. We excluded
the NQ dataset from this evaluation stage as it han-
dles by design any type of query, preventing the
definition of invalid ones.
Evaluation metric To evaluate if systems cor-
rectly handle queries, we utilized F1 score and
3https://huggingface.co/datasets/
anonymousubmission/user-intent-handling
QA IR/E
Setting FIQA FEVER CQA-EN
rec F1 rec F1 rec F1
naïve 100 66.7 100 66.7 100 66.7
enhanced 95.1 95.7 84.4 87.9 94.7 96.6
agentic 97.7 98.8 49.3 64.6 100 99.8
Table 3: User intent handling peformances (recall and
F1) on 500 valid and invalid queries per dataset. The
baseline is a Naïve RAG, where retrieval is performed
for each user query. The Enhanced settings are based
on the semantic router approach, while the Agent au-
tonomously decided whether to use the RAG tool.
recall, to take into account performances on both
the valid and invalid classes.
Results Table 3 reports the results for user in-
tent handling. We found that Agentic slightly
overperforms Enhanced settings in the FIQA and
CDQADupStack-EN tasks. In the case of FEVER,
the former underperforms the latter by a margin
(−28.8 F1 points), due to a very low recall (49.3).
This low recall stems from the system often using
retrieval even in cases it should not. We attribute
these results to the first two datasets having a very
clear domain definition (finance, English grammar),
whereas the FEVER task is much less restrictive by
design, as it aims to verify user queries on factual
information, which makes it harder for the agent
to understand what requests are "valid". We report
the prompts in Appendix A.3.
3.3 Query Rewriting
Definition Much attention has been given to
query rewriting techniques, first introduced by (Ma
et al., 2023). The idea is that when the user query
is tested against the knowledge base for a similarity
QA IR/E
Setting FIQA NQ FEVER CQAD AVG
naïve 45.3 43.7 66.2 45.8 50.3
enhanced 43.5 43.9 81.1 42.8 52.8
agentic 43.2 51.7 83.1 44.3 55.6
Table 4: Query rewriting performances in terms of
NDCG@10. Naïve RAG represents the baseline where
the user query is directly embedded without rewriting.
search, the comparison is often performed among
fairly different texts: the query is usually a short
and dense question, while chunks in the KB can be
long and complex. Query rewriting techniques aim
to reduce this delta by converting the query into
a text with a structure more similar to the target
chunks. Hyde (Gao et al., 2023a), consisting in
substituting the query with a short paragraph that
answers it, has emerged as one of the best perform-
ing techniques (Wang et al., 2024).
Enhanced Implementation We forced the En-
hanced system to perform Hyde query rewriting.
Each user query is automatically rewritten before
retrieval by prompting gpt-4o.
Agentic Implementation We design a prompt
to make the agent aware that rewriting might (Ap-
pendix A.3). Once the Agent has chosen to use the
RAG tool, it can decide to perform this step.
Experimental Settings We run all the queries
in the test sets of the four datasets against the two
systems. In those cases where the Agentic setting
did not perform query rewriting, we calculate the
retrieval metric on the original query.
Evaluation metric All queries in the four
datasets come with annotations on the ground truth
documents they should be linked to. When eval-
uating the quality of the retrieved chunks, we
use the Nomarlized Discounted Cumulative Gain
NDCG@10 (Järvelin and Kekäläinen, 2002). More
details are reported in Appendix A.1.
Results Results in Table 4 show that the Agentic
setting performs better than Enhanced, with an av-
erage gain of +2.8 NDCG@10 points. We attribute
this to the flexibility of the former, which can dy-
namically decide whether to perform rewriting, and
how. Results suggest that query rewriting is ben-
eficial in general, and that an adaptive approach
that decides what to do on a case-by-case basis
is the most effective one. Rewriting is maximally
useful in general when the user query is very dif-
ferent from a question format (FEVER). It can be
observed that when user queries can be of any kind
(NQ), the flexibility of the Agent allows it to out-
perform Enhanced settings (+7.8 points). On the
other hand, for specific-domain settings, they per-
form equally (FIQA). Interestingly, for both IR/E
tasks the Agent outperforms Enhanced settings by
the same delta (+2 and +1.5 points), suggesting
that when RAG is used for information extraction
the flexible rewriting is desirable. Examples are
reported in Appedix C.3.
3.4 Document List Refinement
Definition Previous work has shown how the re-
trieval step may include partially irrelevant or noisy
results, and proposed approaches to improve the
selection process via reranking strategies (Sachan
et al., 2022; Sun et al., 2023; Qin et al., 2024).
Reranking consists in sorting the retrieved chunks,
selecting a subset of highly relevant ones with re-
spect to the user query. Other methods such as
CRAG (Yan et al., 2024) have been proposed, high-
lighting the importance of such step.
Enhanced Implementation We experiment with
this dimension by using an ELECTRA-based
reranker4 (Déjean et al., 2024) on the list of the
20 most similar documents for each user query.
Agentic Implementation The Agentic RAG sys-
tem can inherently attempt to consider a more suit-
able context when needed. Specifically, the agent
may trigger additional retrieval rounds and adapt
the query formulation as it deems appropriate, al-
lowing it to iteratively obtain more relevant context.
We calculate the metric on the last reformulation
of the query that the Agent uses for the RAG tool,
which directly precedes answer generation.
Experimental Settings We run all queries in the
FIQA and CDQStack-En test sets against the two
systems to assess performance on both QA and
IR/E tasks. We did not consider NQ and FEVER
due to their size. As detailed in Section 4, Agentic
RAG would take >7 days on each of their KBs.
Evaluation metric As in query rewriting, we
utilize NDCG@10, leveraging the ground truth
links between queries and documents.
4https://huggingface.co/naver/
trecdl22-crossencoder-electra
QA IR/E
Settings FIQA CQA-EN AVG
naïve 45.0 46.0 45.5
enhanced w/o rewriting 49.0 47.0 48.0
enhanced with rewriting 51.0 48.0 49.5
agent 43.4 44.4 43.9
Table 5: Document list refinement performances in
terms of NDCG@10. Since Agentic retrieval might
indirectly performs query rewriting, we report results
with and without rewriting for the Enhanced setting.
Enhanced and Agentic Implementation To as-
sess this effect, we tested four generators of varying
capability, namely Qwen3-0.6B(without thinking),
Qwen3-4B, Qwen3-8B, and Qwen3-32B(Yang et al.,
2025). We define the generator capability based
on the self-reported performances (Table 6 in Ap-
pendix). We do not aim to assess which model
performs best as a generator, but how the overall
RAG performances are impacted by generators of
different power. We utilise Enhanced RAG with
rewriting and reranking, and the Agentic settings
with the system prompt defined in Appendix A.3.
Experimental Settings We run all queries in the
test sets of FIQA (QA) and CDQStack-En (IR/E)
against the two systems. We did not consider NQ
and FEVER due to their KB size. For both systems,
we set the retrieval chunks number to 5.
Figure 2: Changing underlying LLM performances
(Qwen). For CQADupStack-EN (left), the metric is
based on a pairwise analysis, calculated as the % ratio
of times the larger model’s answer is better than the
smaller counterpart. For FIQA (right), the metric is the
overall % ratio of the classification metric (1 if the an-
swer follows the instruction, 0 otherwise). Both metrics
are calculated via LLM-as-a-Judge (Selene-70B).
Results In the Enhanced setting, re-ranking has
a substantial positive impact on performance. In
contrast, the Agentic RAG setting gains no benefit
from iterating the retrieval step. On average, in
those cases in which the agent decided to perform
again the retrieval (10% of the times), 53% of the
retreived documents remain the same (example re-
ported in Appendix C.2). This highlights that once
the model has taken a decision, it is not likely to
reconsider it. Even when that happens, the process
leads to a second retrieval step that only in one case
out of two modifies the retrieved documents with
respect to the previous step.
3.5 Underlying LLM
Definition Both Agentic and Enhanced settings
are highly impacted by the choice of the under-
lying LLM, as different models produce different
answers even when provided with the same query
and retrieved context. Furthermore, the role of the
generator is particularly critical in Agentic RAG,
where the model must not only produce the final
answer but also make decisions at each stage of
the workflow. We are interested in quantifying the
impact that a weaker generator has on the system,
compared to what a stronger one would have.
Evaluation metric We evaluate the quality of
the final answers generated by both systems using
automatic metrics, employing the LLM-as-a-judge
paradigm. Out of the many approaches proposed in
the literature (Kim et al., 2024; Wang et al., 2025;
Es et al., 2024) we select Selene-70B (Alexandru
et al., 2025), a fine-tuned version of Llama-3.3-
70B-Instruct, as its smaller version scores among
the top ones in the LLM-as-a-Judge (AtlaAI, 2025).
Furthermore, we analyzed its reliability in evaluat-
ing our two selected datasets. First, Selene judg-
ments has been shown to closely match human eval-
uations on financial QA tasks by its authors, which
is relevant for the FIQA dataset. Therefore, we
could adopt the binomial 0–1 classification metric
for this dataset, shown to be most human-aligned
option. On the other hand, for CQADupStack-En,
such alignment has not been demonstrated. There-
fore, we performed manual annotation on a subset
of the testing data (5%, 312 answer pairs in total),
and calculated the agreement rate between the two
human annotators and the automatic metric. We
select the pairwise metric (given two answers of
two models, select the best one). Inter-annotator
agreement rate (ratio of times they both chose A
or B) is 71.9%, while the human-model agreement
is 65.4%. Manual annotation took an average of
1.5 minutes per pair, resulting in 15.5 hours over-
all. The annotation guidelines are reported in the
Appendix. To summarize the impact of underlying
model changes, we calculate the ratio of times the
larger model wins over the smaller counterpart.
Results We aim to assess whether Enhanced and
Agentic systems exhibit distinct performance pat-
terns across model scales. Figure 2 reports the
resulting average scores for both datasets, which
show that the two systems do not present significant
differences in patterns when changing the under-
lying LLM. In fact, the performance increase in
FIQA follows the same distribution both Enhanced
and Agentic RAG, and the same is true for the ratio
of times in which the bigger model is preferred
over the smaller one in CQADupStack-En settings.
Full results are reported in Appendix A.5.
4 Cost and Time
Fixed costs for Enhanced and Agentic settings
The two systems share some fixed costs due to hard-
ware requirements. We utilize for both settings
a t3.large ec2 AWS instance (0.09 $/h per on-
demand usage) to instantiate a relational database
with vector search capabilities (pgvector5). We
implement the RAG application backend on a
t2.medium (0.05 $/h). We test open LLMs in a
proprietary 8×a40 cluster (46GB). We run Qwen3
0.6B, 4B, and 8B on a single gpu, while the 32B
version on 4× a40. A similar setting on AWS can
be g4ad.8xlarge, which costs 1.9 $/h. The time
and cost related to retrieval are the same in both set-
tings. We use OpenAI text-embedding-3-small
as default embedder model with cosine similar-
ity. Enhanced RAG does re-ranking with a 300M-
parameter model on the same cluster as the LLMs,
with a negligible added cost.
Runtime costs and number of tokens We ap-
proximate the runtime cost of RAG systems by
the number of processed input and output tokens.
This hardware-agnostic metric enables cost esti-
mation across different deployment environments
given on model throughput and hourly pricing. We
analyse token usage for both GPT and Qwen mod-
els to quantify cost differences between Enhanced
and Agentic settings. We also report end-to-end
latency—the time from receiving a query to return-
ing the final answer. Table 9 (Appendix) reports a
summary of time and tokens per approach.
Overall, we find that Agentic settings are more ex-
pensive and more time-consuming than Ehanced
settings, requiring an average of 3.3× more in-
put tokens and 1.9× more output tokens among
datasets, as well as 1.5× more time.
5https://github.com/pgvector/pgvector
For valid queries in Enhanced RAG, we find that
roughly 45–50% of the total time is spent gener-
ating the answer, a similar proportion is spent on
query rewriting, 0–5% on retrieval, and 0–2% on
document re-ranking. The dominant factor in la-
tency is the LLM calls. Therefore, any performance
optimization should focus primarily on that.
5 Conclusion
Our experimental comparison reveals that neither
Enhanced nor Agentic RAG is universally superior.
First, we observe that in well-defined domains with
highly structured user behavior, Agentic RAG ex-
cels at handling user intent, thanks to its ability to
understand the user query. However, in broader or
noisier domains, our Enhanced RAG routing sys-
tem proves more reliable. textcolorblueDevelopers
should consider using the Agentic approach when
possible, considering it does not require any manu-
ally crafted example to run. Second, with respect
to alignment of the query to the structure and se-
mantics of the documents in the KB, Agentic RAG
outperforms Enhanced RAG retrieval quality. Its
dynamic use of query rewriting allows for retrieval
of more relevant context. Third, we found that
when Agentic RAG selects certain documents, it is
not as good as the re-ranking done by Enhanced
RAG at selecting just the most meaningful docs.
Fourth, we observe that changing the underlying
LLM produce the same changes in performance
in both settings: as the LLM becomes larger, per-
formance improves at comparable rates. Our cost
analysis highlights that Agentic RAG is systemati-
cally more expensive—up to 3.6 times more—due
to additional reasoning steps and repeated tool calls.
This cost difference should not be overlooked: a
well-optimized Enhanced RAG can match or ex-
ceed Agentic performance while remaining more
efficient.
In summary, developers should consider combin-
ing the two approaches to achieve the best per-
formances, taking into accout the increase in cost.
The Agentic approach suits best user-intent routing
(even without any manually crafted examples) and
query rewriting. On the other hand, our results sug-
gest that integrating an explicit re-ranking step into
Agentic pipelines could provide substantial gains.
References
AIME. 2024. Aime problems and solutions.
https://artofproblemsolving.com/wiki/
index.php/AIME_Problems_and_Solutions.
Accessed 2024.
Andrei Alexandru, Antonia Calvi, Henry Broomfield,
Jackson Golden, Kyle Dai, Mathias Leys, Maurice
Burger, Max Bartolo, Roman Engeler, Sashank Pisu-
pati, Toby Drane, and Young Sun Park. 2025. Atla
selene mini: A general purpose evaluation model.
Preprint, arXiv:2501.17195.
Salaheddin Alzubi, Creston Brooks, Purva Chiniya,
Edoardo Contente, Chiara von Gerlach, Lucas Ir-
win, Yihan Jiang, Arda Kaz, Windsor Nguyen, Se-
woong Oh, Himanshu Tyagi, and Pramod Viswanath.
2025. Open deep search: Democratizing search
with open-source reasoning agents. Preprint,
arXiv:2503.20201.
Muhammad Arslan, Hussam Ghanem, Saba Munawar,
and Christophe Cruz. 2024. A survey on rag with
llms. Procedia Computer Science, 246:3781–3790.
28th International Conference on Knowledge Based
and Intelligent information and Engineering Systems
(KES 2024).
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
In The Twelfth International Conference on Learning
Representations.
AtlaAI. 2025. AtlaAI/judge-arena: Benchmarking llms
as evaluators. https://huggingface.co/spaces/
AtlaAI/judge-arena. Accessed: 2025-11-18.
Contributors Aurelio Labs. 2025. aurelio-
labs/semantic-router: Framework for orchestrating
role-playing, autonomous ai agents. https:
//github.com/aurelio-labs/semantic-router.
Accessed: 2025-11-18.
AWS. 2025. Amazon web services — bedrock. https:
//aws.amazon.com/bedrock/.
Azure. 2025. Azure ai search — rag solution tu-
torial. https://docs.azure.cn/en-us/search/
tutorial-rag-build-solution.
BrainBlend-AI. 2025. Brainblend-ai/atomic-agents:
Modular agents framework for building agents
from atomic components. https://github.com/
BrainBlend-AI/atomic-agents. MIT License; ac-
cessed: 2025-11-18.
Kaitong Cai, Jusheng Zhang, Yijia Fan, Jing Yang, and
Keze Wang. 2026. Racot: Plug-and-play contrastive
example generation mechanism for enhanced llm rea-
soning reliability. Proceedings of the AAAI Confer-
ence on Artificial Intelligence, 40(36):30112–30120.
Andrew Chen, Andy Chow, Aaron Davidson, Arjun
DCunha, Ali Ghodsi, Sue Ann Hong, Andy Konwin-
ski, Clemens Mewald, Siddharth Murching, Tomas
Nykodym, Paul Ogilvie, Mani Parkhe, Avesh Singh,
Fen Xie, Matei Zaharia, Richard Zang, Juntai Zheng,
and Corey Zumar. 2020. Developments in mlflow: A
system to accelerate the machine learning lifecycle.
In Proceedings of the Fourth International Workshop
on Data Management for End-to-End Machine Learn-
ing, DEEM ’20, New York, NY, USA. Association
for Computing Machinery.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. Proceedings of
the AAAI Conference on Artificial Intelligence,
38(16):17754–17762.
Y. Chen, B. Gu, Y. Qu, Y. Chen, L. Cui, and L. Gao.
2026. Insight-rag: Internal state signals-heightened
trustworthy retrieval-augmented generation. In Neu-
ral Information Processing (ICONIP 2025), volume
2754 of Communications in Computer and Informa-
tion Science, Singapore. Springer.
CrewAIInc Contributors. 2025a. crewAIInc/cre-
wAI: Framework for orchestrating role-playing,
autonomous ai agents. https://github.com/
crewAIInc/crewAI. Accessed: 2025-11-18.
Pydantic Contributors. 2025b. pydantic/pydantic-ai:
Genai agent framework, the pydantic way. https://
github.com/pydantic/pydantic-ai. Accessed:
2025-11-18.
Gustavo de Aquino e Aquino, Nádila da Silva de
Azevedo, Leandro Youiti Silva Okimoto, Leonardo
Yuto Suzuki Camelo, Hendrio Luis de Souza Bra-
gança, Rubens Fernandes, Andre Printes, Fábio Car-
doso, Raimundo Gomes, and Israel Gondres Torné.
2025. From rag to multi-agent systems: A survey of
modern approaches in llm development. Preprints.
Deloitte. 2024. The state of generative ai in the enter-
prise. https://www2.deloitte.com.
Hervé Déjean, Stéphane Clinchant, and Thibault For-
mal. 2024. A thorough comparison of cross-
encoders and llms for reranking splade. Preprint,
arXiv:2403.10407.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. In Proceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, KDD ’24,
page 6491–6501, New York, NY, USA. Association
for Computing Machinery.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023a. Precise zero-shot dense retrieval without rel-
evance labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025a. Search-o1: Agentic search-enhanced
large reasoning models. Preprint, arXiv:2501.05366.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and
Haofen Wang. 2023b. Retrieval-augmented genera-
tion for large language models: A survey. Preprint,
arXiv:2312.10997.
Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh
Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei,
Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit
Chan, Yankai Chen, Zhongfen Deng, Yinghui Li,
Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming
Zhang, Yangqiu Song, and Philip S. Yu. 2025b.
Giulia Rutigliano. 2023. Creating flamel:
Towards agentic rag with deep reasoning: A sur-
A journey of ai and design collaboration.
vey of rag-reasoning systems in llms. Preprint,
https://medium.com/design-group-italia/
arXiv:2507.09477.
creating-flamel-a-journey-of-ai-and-design-collaboration-a38397a8d353.
Jerry Liu. 2022. LlamaIndex.
Doris Hoogeveen, Karin M. Verspoor, and Timothy
Baldwin. 2015. Cqadupstack: A benchmark data
set for community question-answering research. In
Proceedings of the 20th Australasian Document Com-
puting Symposium, ADCS ’15, New York, NY, USA.
Association for Computing Machinery.
Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models. Preprint, arXiv:2404.10981.
IBM. 2025. Ibm watsonx — rag develop-
ment. https://www.ibm.com/it-it/products/
watsonx-ai/rag-development.
Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumu-
lated gain-based evaluation of ir techniques. ACM
Trans. Inf. Syst., 20(4):422–446.
Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang,
Shayne Longpre, Hwaran Lee, Sangdoo Yun,
Seongjin Shin, Sungdong Kim, James Thorne, and
Minjoon Seo. 2024. Prometheus: Inducing fine-
grained evaluation capability in language models.
Preprint, arXiv:2310.08491.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 5303–5315, Singa-
pore. Association for Computational Linguistics.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: iterative
refinement with self-feedback. In Proceedings of the
37th International Conference on Neural Information
Processing Systems, NIPS ’23, Red Hook, NY, USA.
Curran Associates Inc.
Macedo Maia, Siegfried Handschuh, André Freitas,
Brian Davis, Ross McDermott, Manel Zarrouk, and
Alexandra Balahur. 2018. Www’18 open challenge:
Financial opinion mining and question answering. In
Companion Proceedings of the The Web Conference
2018, WWW ’18, page 1941–1942, Republic and
Canton of Geneva, CHE. International World Wide
Web Conferences Steering Committee.
Tula Masterman, Sandi Besen, Mason Sawtell, and Alex
Chao. 2024. The landscape of emerging ai agent
architectures for reasoning, planning, and tool calling:
A survey. Preprint, arXiv:2404.11584.
McKinsey. 2023. The economic potential
of generative ai: The next productivity
frontier. https://www.mckinsey.com/
capabilities/tech-and-ai/our-insights/
LangChain Inc. 2025. langchain-ai/langgraph: Build re-
the-economic-potential-of-generative-ai-the-next-produc
silient language agents as graphs. https://github.
com/langchain-ai/langgraph. MIT License; ac-
cessed: 2025-11-18.
Microsoft. 2025. microsoft/autogen: Autogen – agent
orchestration and tool calling. https://github.
com/microsoft/autogen. MIT License; accessed:
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
2025-11-18.
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
Fnu Neha and Deepshikha Bhati. 2025. Traditional
rag vs. agentic rag: A comparative study of retrieval-
augmented systems. Authorea Preprints.
intensive nlp tasks. In Advances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang,
Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu
Liu, Donald Metzler, Xuanhui Wang, and Michael
Bendersky. 2024. Large language models are effec-
tive text rankers with pairwise ranking prompting. In
Findings of the Association for Computational Lin-
guistics: NAACL 2024, pages 1504–1518, Mexico
City, Mexico. Association for Computational Lin-
guistics.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
Gpqa: A graduate-level google-proof q&a bench-
mark. Preprint, arXiv:2311.12022.
Aymeric Roucher, Albert Villanova del Moral, Thomas
Wolf, Leandro von Werra, and Erik Kaunismäki.
2025. ‘smolagents‘: a smol library to build
great agentic systems. https://github.com/
huggingface/smolagents.
Devendra Sachan, Mike Lewis, Mandar Joshi, Armen
Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke
Zettlemoyer. 2022. Improving passage retrieval with
zero-shot question generation. In Proceedings of
the 2022 Conference on Empirical Methods in Nat-
ural Language Processing, pages 3781–3797, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: language agents with verbal reinforcement
learning. In Proceedings of the 37th International
Conference on Neural Information Processing Sys-
tems, NIPS ’23, Red Hook, NY, USA. Curran Asso-
ciates Inc.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag. Preprint,
arXiv:2501.09136.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT good at search?
investigating large language models as re-ranking
agents. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 14918–14937, Singapore. Association for
Computational Linguistics.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. In Proceedings of
the Neural Information Processing Systems Track on
Datasets and Benchmarks, volume 1.
the-pocket. 2025. the-pocket/pocketflow: Llm
framework in 100 lines. https://github.com/
the-pocket/PocketFlow. MIT License; accessed:
2025-11-18.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. In Proceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
PeiFeng Wang, Austin Xu, Yilun Zhou, Caiming Xiong,
and Shafiq Joty. 2025. Direct judgement preference
optimization. In Proceedings of the 2025 Conference
on Empirical Methods in Natural Language Process-
ing, pages 1979–2009, Suzhou, China. Association
for Computational Linguistics.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Huang. 2024. Searching for best practices in
retrieval-augmented generation. In Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pages 17716–17736, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao,
Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Wei-
wen Liu, Yasheng Wang, Ruiming Tang, Weinan
Zhang, and Yong Yu. 2025. Infodeepseek:
Benchmarking agentic information seeking for
retrieval-augmented generation. arXiv preprint
arXiv:2505.15872.
Anbang Xu, Tan Yu, Min Du, Pritam Gundecha, Yufan
Guo, Xinliang Zhu, May Wang, Ping Li, and Xinyun
Chen. 2024. Generative ai and retrieval-augmented
generation (rag) systems for enterprise. In Proceed-
ings of the 33rd ACM International Conference on In-
formation and Knowledge Management, CIKM ’24,
page 5599–5602, New York, NY, USA. Association
for Computing Machinery.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
CoRR, abs/2401.15884.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report. Preprint,
arXiv:2505.09388.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024. Crag - comprehensive rag bench-
mark. In Advances in Neural Information Processing
Systems, volume 37, pages 10470–10490. Curran As-
sociates, Inc.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. In International Conference on Learning
Representations (ICLR).
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and
Le Hou. 2023. Instruction-following evaluation for
large language models. Preprint, arXiv:2311.07911.
A Appendix
A.1 NDCG metric
When evaluating the quality of the retrieved chunks,
we use the Nomarlized Discounted Cumulative
Gain NDCG@10 (Järvelin and Kekäläinen, 2002).
NDCG stands for Nomarlized Discounted Cumu-
lative Gain (Järvelin and Kekäläinen, 2002) and is
one of the most common metrics used to assess the
effectiveness of a ranking model. NDCG at cutoff
K is defined as:
NDCG@K ≡
DCG@K
maxDCG@K (1)
where maxDCG@K is the maximum DCG@K that
can be obtained from the given relevance labels,
and where DCG@K is defined as:
DCG@K ≡
K
i=1
2li − 1
log(1 + i) (2)
where li is the relevance label (the ground truth la-
bel) of the document in position i in the rank. Since
Equation A.1 is always positive, Equation A.1
is a number bounded between 0 and 1, where
NDCG@K equal to 1 means that we have a perfect
ranked list.
A.2 Invalid query generation
For each dataset, the invalid queries are generated
by prompting gpt-4o with the following prompt:
You are a data augmentation assistant focused
on GENERATING OFF-DOMAIN QUERIES.
You are given VALID EXAMPLES representing
the domain of a knowledge base (KB).
Produce EXACTLY n distinct user queries
that are OFF-TOPIC relative to the examples,
thus NOT answerable by retrieval over that KB.
REQUIREMENTS:
- Each query 5-18 words, self-contained,
concrete.
- Benign topics only; no sensitive or
disallowed content.
- Must NOT overlap semantically with examples
or with each other.
- Avoid meta references (no mentions of
’examples’, ’dataset’, ’KB’, ’instructions’).
- Diverse domains (finance, cooking, travel,
biology, art, sports, etc.).
OUTPUT STRICT JSON ONLY (no extra text):
{{"invalid_queries": ["query 1", "query 2",
... (total 250) ]}}
VALID EXAMPLES:
{examples_formatted}
Return ONLY the JSON now.
To quantify the distances between the two
group of queries ("VALID" and "INVALID")
for each dataset, we calculate the cosine sim-
ilarity between embeddings calculated using
Qwen/Qwen3-Embedding-0.6B:
• between each valid query and the other 249
valid queries (V-V),
• between each valid query and the other 250
invalid queries (V-I).
Higher deltas between V-V and V-I values mean
that queries are quite distant, as desired. We aver-
age the similarities V-V and V-I and found good dis-
crimination on CQAD-EN (V-V 0.449, V-I 0.276),
NQ (V-V 0.339, V-I 0.266), FIQA NQ (V-V 0.404,
V-I 0.257). This distance is much lower for FEVER,
which indeed we excluded (V-V 0.289, V-I 0.225).
A.3 Prompts
A.3.1 Agentic RAG
The Agentic RAG is defined by three nodes: the
orchestrator, the answer node, and the RAG node.
Here we report the orchestrator prompt:
Context
You are a powerful Al agent. Your task is to
{task_description}
Inputs
•query: {query}
•previous_tool_results:
{previous_tool_results}
Guidelines To do complete your task,
you can take several actions:
• answer: {answer_node_description}
• document_retrieval:
{retrieval_node_description}
Important
• Don’t repeat actions unnecessarily.
• When composing the query for RAG search you
must always write a passage to answer the
input query provided by the user.
• If you think that the retrieved documents
in the previous_tool_results can be further
improved, you can rewrite the query and call
document_retrieval again.
• If you see the relevant retrieved documents
already in the previous_tool_results, you
should proceed with answering!
Metadata:
• current reasoning step:
{current_reasoning_step}
• maximum reasoning steps:
{max_reasoning_steps}
For each evaluation dataset, the prompt if filled
with the following:
FIQA
task_description:
help users on financial issues using the
tools at your disposal.
answer_node_description:
provide the final answer to the user,
utilizing the retrieved documents if
necessary. Choose this action when you
have gathered enough information from the
previous_tool_results.
retrieval_node_description:
retrieve documents from a knowledge base of
financial information. Choose this to ground
answers on external sources
NQ
task_description:
help users using the tools at your disposal.
answer_node_description:
provide the final answer to the user,
utilizing the retrieved documents if
necessary. Choose this action when you
have gathered enough information from the
previous_tool_results.
retrieval_node_description:
retrieve documents from a knowledge base of
potentially useful information. Choose this
to ground answers on external sources.
FEVER
task_description:
help users verifying their queries on
factual knowledge. For each user claim, you
determine if it is supported or refuted by
the world-knowledge on the topic.
answer_node_description:
provide the final answer to the user,
utilizing the retrieved documents if
necessary. Choose this action when you
have gathered enough information from the
previous_tool_results.
retrieval_node_description:
retrieve documents from a knowledge base of
factual information. Choose this to ground
answers on gold-standard knowledge. Be
careful, your knowledge might be outdated.
CQADUPSTACK-EN
task_description:
help users finding blog posts about English
grammar issues that are related to the
user query. You have to find the relevant
documents, summarize them, and inform the
user about them.
answer_node_description:
provide a summary of the retrieved blog posts
to the user. Choose this action when you have
gathered enough information on grammar blog
posts from the previous_tool_results.
retrieval_node_description:
retrieve blog posts from a knowledge base of
English grammar issues. Choose this to find
relevant blog posts. Never use it for queries
not related to English grammar issues.
The answer node is defined as follows:
FIOA & NQ
Context:
You are a question-answering assistant. You
are provided the original user query, a list
of tools called and their results, a draft
answer, and the conversation history. Provide
the final answer to the user.
Inputs:
question: {query}
previous_tool_results:
{previous_tool_results}
draft_answer: {draft_answer}
Guidelines:
- be consistent with previous responses in
the conversation.
- if a follow-up question references a
previous conversation, use the conversation
history to provide context.
FEVER
Context:
You are a question-answering assistant. You
are provided the original user query, a
list of tools called and their results, a
draft answer, and the conversation history.
Provide a summary of the grounding reasons
and evidence.
Inputs:
question: {query}
previous_tool_results:
{previous_tool_results}
draft_answer: {draft_answer}
Guidelines:
- be consistent with previous responses in
the conversation.
- if a follow-up question references a
previous conversation, use the conversation
history to provide context.
- summarize the reasons that support or
contradict the statement.
CQADUPSTACK-EN
Context:
You are a blog posts summarizer. You are
provided the original user query, a list
of tools called and their results, a draft
answer, and the conversation history. Provide
a bullet point list of the relevant blog posts
related to the user query, with one bullet
point per blog post. Each bullet point must
include: i) a short title (max 10 words); ii)
a one-sentence summary (max 15 words) of the
blog post; iii) the reference to the document.
You do not answer the question. Instead, you
provide an overview of the retrieved blog
posts to the user. Your objective is to make
the user aware of the relevant blog posts
related to their query.
Inputs:
question: {query}
previous_tool_results:
{previous_tool_results}
draft_answer: {draft_answer}
Guidelines:
- be consistent with previous responses in
the conversation.
- introduce the bullet point list with a short
sentence such as “Here are some blog posts
that might be relevant to your query:”.
- do not answer the question directly. Your
goal is to summarize the retrieved blog posts.
The RAG node does not have a specific system
prompt.
Query rewriting Query rewriting is performed
with the following prompt: "Convert the user
query into a {type_of_doc}", where type_of_doc
differs based on the dataset ("longer blog post" for
CQADupStack, "passage to answer it" for FiQA
and NQ, "longer factual statement" for FEVER).
A.3.2 Enhanced RAG
For each of the four evaluation dataset a different
system prompt is defined:
FIQA
You are an advanced AI assistant expert in
financial topics. You are provided with a
question from a user and some potentially
useful context. You are able to use the useful
context to further inform your responses.
This context is divided into chunks drawn
from different documents. Remember that the
useful context is your cheatsheet, so you
must always avoid mentioning it to the user.
Never mention the context.
**Very Important**:
- You MUST NEVER invent documents that are not
mentioned in the ’useful context’.
- if you do not see a chunk matching the
user’s question, do not mention it.
- avoid hallucinating data. Answer truthfully
and forthrightly, stating only what you can
derive from the provided context or from
general knowledge.
NQ
You are an advanced AI assistant. You are
provided with a query from a user and some
potentially useful context. You are able to
use the useful context to further inform your
responses. This context is divided into chunks
drawn from different documents. Remember that
the useful context is your cheatsheet, so you
must always avoid mentioning it to the user.
Never mention the context.
**Very Important**:
- you must never invent documents that are not
mentioned in the useful context.
- if you do not see a chunk matching the
user’s question, do not mention it.
- avoid hallucinating data. Answer truthfully
and forthrightly, stating only what you can
derive from the provided context or from
general knowledge.
FEVER
You are an advanced AI assistant. Your task
is to help the user verify their queries on
factual knowledge. For each user claim, you
determine whether it is supported or refuted
by the world-knowledge on the topic. You
are provided with a query from the user and
some potentially useful context. You are
able to use the useful context to further
inform your responses. This context is
divided into chunks drawn from different
documents. Remember that the useful context
is your cheatsheet, so you must always avoid
mentioning it to the user. Never mention the
context.
**Very Important**:
- you must never invent documents that are not
mentioned in the useful context.
- if you do not see a chunk matching the
user’s question, do not mention it.
- avoid hallucinating data. Answer truthfully
and forthrightly, stating only what you can
derive from the provided context or from
general knowledge.
CQADUPSTACK-EN
You are a blog posts summarizer. Your task is
to help users find blog posts about English
grammar issues that are related to the
user query. You are provided with relevant
documents; summarize them and inform the user
about them. Provide a bullet point list of
the relevant blog posts related to the user
query, with one bullet point per blog post.
Each bullet point must include: i) a short
title (max 10 words); ii) a one-sentence
summary (max 15 words) of the blog post; iii)
the reference to the document. You do not
answer the question. Instead, you provide
an overview of the retrieved blog posts to
the user. Your objective is to make the user
aware of the relevant blog posts related to
their query.
**Very Important**:
-be consistent with previous responses in the
conversation.
- introduce the bullet point list with a short
sentence such as “Here are some blog posts
that might be relevant to your query:”.
- do not answer the question directly. Your
goal is to summarize the retrieved blog posts.
Query rewriting Query rewriting is performed
with the following prompt: Please write a pas-
sage to answer the question. \n Question:
{user_query}\n Passage:".
A.4 Routing system details
The schema of the routing system we implement
for Enhanced RAG settings is described in Figure 3.
The routing mechanism relies on example-based
classification. Queries are compared to two ref-
erence sets: valid and invalid. If a query is suf-
ficiently similar to the valid set, it is processed;
otherwise, it is rejected. The key challenge is de-
termining what “sufficiently similar” means, i.e.,
selecting an appropriate similarity threshold.
Threshold selection Threshold selection can be
approached in multiple ways (e.g., random search,
linear models, classification algorithms). If class
definitions are clear and well-separated, threshold
tuning becomes less critical, since queries will nat-
urally cluster around their correct class. In prac-
tice, however, class boundaries often contain noise,
making the threshold an essential safeguard against
misclassification.
A.5 Underlying LLM evaluation metrics
The results for the metrics calculated by means of
Selene-70B are reported in Table 7 (FIQA) and
Figure 4 (CQADupStack-EN). Here we report the
evaluation guidelines for CQADupStack-English
used by human annotators:
GENERAL DESCRIPTION
You are evaluating a system that, given a user
query, retrieves a list of similar queries
previously submitted by other users. These
past queries are called documents. The system
output must be a list of bullet points, each
representing one and only one document. An
ideal bullet point contains:
- a title of at most five words,
- a brief description of the retrieved
document,
- a reference to the document number.
Figure 3: Schema of the routing system utilized for
Enhanced. Two query classes are defined—valid and
invalid—each represented by embedded examples from
a validation set. For each incoming query, the router em-
beds it, retrieves the top-20 most similar examples via
cosine similarity, and selects the class with the highest
average similarity. If the mean similarity of the selected
class exceeds a predefined threshold, that label is as-
signed; otherwise, the router returns None, which can
be further remapped according to business logic.
Important: the documents do not contain
answers to the user’s query; they are actual
queries submitted in the past. If the system
response includes a document description that
explains the grammar question rather than
restating the query itself, this is incorrect.
ANNOTATION:
For each query, there are two system responses,
A and B. Your task is to select the better
response based on the criteria above. If you
are unsure, you may write “I don’t know.” If
both responses are of equal quality, write
“same.”
Common situations include:
- the document description contains an answer
rather than the original query (serious
error).
- one of the listed documents is irrelevant
to the query (serious error).
- if the two responses differ in the number
of documents and all documents are relevant,
always choose the one with more documents.
- the response mixes the document number with
the title (error).
- the document description is excessively long
(error).
- the document number reference is missing
(error).
Prompt used for Selene 70B for CQADupStack-
English, based on the one released with the original
paper (Alexandru et al., 2025):
EVALUATION TASK
You are a helpful assistant evaluating the
quality of responses for a given instruction.
Your goal is to select the best response
for the given instruction. Two responses
are provided, Response A and Response B,
generated by two different AI chatbots. You
must select either Response A or Response B.
Do not state that both or neither are good.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether
the response satisfies the provided rubric.
Then consider its helpfulness, accuracy,
level of detail, harmlessness, etc.
(2) Responses should NOT contain more/less
than what the instruction asks for, as
such responses do NOT precisely execute the
instruction.
(3) You should avoid any potential bias
and your judgment should be as objective as
possible. Here are some potential sources of
bias:
- The order in which the responses were
presented should NOT affect your judgment,
as Response A and Response B are **equally
likely** to be the better.
- The length of the responses should NOT
affect your judgement, as a longer response
does not necessarily correspond to a better
response. When making your decision, evaluate
if the response length is appropriate for the
given instruction.
Your reply should strictly follow this format:
**Reasoning:** <feedback evaluating the
responses>
**Result:** <A or B>
Here is the data.
Instruction: {full_prompt_for_answer_gene}
Response A: {assistant_response_a}
Response B: {assistant_response_b}
Score Rubrics: [Which of the following
responses better handles the Instruction - A
or B?]
Model Qwen3 General Align Reason GPQA-D IFEVAL AIME ’24
AVG
0.6B 22.9 54.5 3.4 26.9
4B 55.9 81.9 73.0 70.3
8B 62.0 85.0 76.0 74.3
32B 68.4 85 81.4 78.3
Table 6: Performances of the models selected to analyse
the impact of the underlying LLM as reported by Yang
et al. (2025). The selected benchmarks are Graduate-
Level Google-Proof QA Diamond (Rein et al., 2023),
Instruction Following Eval (Zhou et al., 2023), and
the American Invitational Mathematics Examination
(AIME, 2024).
**Reasoning:** <Your feedback>
**Result:** <Yes or No>
Here is the data:
Instruction: {full_prompt_for_answer_gen}
Response:
{assistant_response}
Score Rubrics:
Does the response correctly and completely
answers the question posed in the instruction?
Answer Yes or No.
Yes: The response accurately and completely
addresses the question posed in the
instruction
No: The response fails to accurately and
completely address the question posed in the
instruction.
B Limitations
Prompt used for Selene 70B for FIQA:
You are tasked with evaluating a response
based on a given user input and binary
scoring rubric that serves as the evaluation
standard. Provide comprehensive feedback on
the response quality strictly adhering to the
scoring rubric, followed by a binary Yes/No
judgment. Avoid generating any additional
opening, closing, or explanations.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether
the response satisfies the provided rubric.
The basis of your score should depend exactly
on the rubric. However, the response does
not need to explicitly address points raised
in the rubric. Rather, evaluate the response
based on the criteria outlined in the rubric.
Your reply must strictly follow this
format:
This study has some limitations that should be con-
sidered when interpreting the results. Although
our evaluation covers key dimensions of RAG be-
havior, document summarization and document
repacking (re-sorting documents in the context ac-
cording to their importance) have not been con-
sidered. Furthermore, we restricted our analysis
to Agentic systems focusing on pure RAG given
our interest in this type of industrial applications.
Therefore, our agent is only equipped with a single
tool. This choice restricts our analysis to the scope
of this work, and falls shorts in providing insides
in Agents performing tasks other than RAG.
C Examples
Here we report some examples of systems behavior.
Framework pros cons
smolagents minimal; codeagent lack of low-level abstraction; lower control and security
issues because of codeagent (can be handled)
langgraph graph abstraction; many integrations; many re-
sources for implementing popular patterns
llamaindex many integrations; many resources for imple-
menting popular patterns
high-complexity codebase
pocketflow minimal; graph abstraction need to perform implementation
crewAI focus on multi-agents
autogen mature ecosystem high complexity
pydanticAI type safety; Python oriented early stages of development; requires learning some
concepts
atomic agents modular early stages of development, limited documentation
Table 8: Comparison of frameworks considered for Agentic RAG implementation. The reported “pros” and
“cons” are defined within the scope of this work—namely, the construction of an agent equipped with a single
RAG tool—and should not be considered exhaustive or universally applicable. We considered SmolAgents
(Roucher et al., 2025), LangGraph (LangChain Inc., 2025), LlamaIndex (Liu, 2022), PocketFlow (the-pocket, 2025),
CrewA (Contributors, 2025a), AutoGen (Microsoft, 2025), PydanticAI (Contributors, 2025b), and Atomic Agents
(BrainBlend-AI, 2025), offering different advantages and reflecting different design choices. For this work, we
select PocketFlow, a lightweight framework that offers a simple graph-based abstraction.
FIQA CQAD-EN
Model time tot tokens ratio (Ag/En) time tot tokens ratio (Ag/En)
s input output time input output s input output time input output
GPT-4.1-nano En 9.0 1683 465 1.1 2.2 0.8 7.0 856 331 1.2 3.0 0.9
10,2 3676 348 8.6 2463 297
Ag Qwen3-0.6B En Ag 8.1 1743 236 22.1 4978 979 2.2 2.9 4.1 8.1 862 254 8.9 3032 867
1.1 3.5 3.4
Qwen3-4B En Ag 35.5 1743 1435 38.6 4834 1704 1.1 2.8 1.2 31.0 862 1019 37.2 3372 1943
1.2 3.9 1.9
Qwen3-8B En Ag 58.5 1743 1490 69.9 4943 1837 1.2 2.8 1.2 58.4 862 1339 54.3 3394 1983
0.9 3.9 1.5
Qwen3-32B En Ag 62.6 1743 1695 93.8 4766 1866 1.5 2.7 1.1 43.9 862 1109 101.9 3359 1636
2.3 3.9 1.5
AVG ratio 1.5 2.7 1.7 1.4 3.6 1.8
Table 9: Analysis of costs (measured by number of input and output tokens) and time (end-to-end latency experienced
by the user when running a query). The "ratio" columns represent the multiplicative factor to go from the Enahnced
to the Agentic settings, e.g. a ratio of 1.5 means that the Agent is 50% more expensive than the Enhanced setting.
Qwen3-0.6B is run without thinking mode, resulting in substantially shorter outputs compared to the larger Qwen
models. Agentic RAG always performed a maximum of 3 turns. In scenarios requiring more turns, tokens consumed
by the agent would increase.
FIQA
Qwen enhanced agentic
0.6B 51,4 58,4
4B 90,9 87,3
8B 93,7 91
32B 95,5 91,8
Table 7: Classification metric based on Selene-70B for
FIQA.
Figure 4: Pairwise metric based on Selene-70B for
CQADupStack-EN.
C.1 Agent workflow
USER QUERY: Where should I park my rainy-day
/ emergency fund?
AGENT REASONING:
To provide a well-informed answer about where
to store an emergency fund, I need to retrieve
authoritative financial guidance on safe,
liquid, and accessible options for emergency
savings.
QUERY SENT FOR RETRIEVAL:
Best places to keep emergency fund for safety
and liquidity
RETRIEVED DOCUMENTS:
DOCUMENT 1 I would suggest your local
credit union or local bank for security and
liquidity. Liquidity is probably the most
important issue for a emergency fund.
DOCUMENT 2 As others have noted, you can do
better than a checking or savings account. If
youŕe going to invest emergency money, the
vehicle you put it into should be: Liquid -
Wherever you put it, you should be able to
quickly cash it out. Highly liquid exchange
traded products are good for this. Low
volatility/drawdowns - If you need at least 6
months of your paycheck to cover you in the
event of an emergency, you don´ t want to park
it in a portfolio that can potentially lose
30% value. Insured - Your investments should
have SIPC coverage (protection against losses
resulting from failure on part of broker).
Moderate/Steady Growth - If the emergency
fund doesn´ t grow, youĺl need to continually
pump money into it. My śteady growth´ portfolio
is majorly allocated to fixed income. Within
that, a major portion is allocated to high
yielding instruments. Over the past 10 years,
itś seen at least a 7% annualized return.
DOCUMENT 3 Firstly well done on building a
really sold base of savings. An emergency
fund needs to have two key characteristics: Be
enough to get you through a typical emergency
event (often seen as approx. 6 months’ salary
in your style of situation assuming you have
no dependents etc) Be liquid and available
to you instantly if an emergency arises Once
you have decided how much you will need for
1), you then generally find the best interest
available on an instant access savings account
and leave it there. Itś important to note
that because you need it very liquid and very
secure you will basically never make (nor
should you expect to make) any sizeable rate
of interest on your emergency fund. Once this
is done, whatever left should be invested in
an asset/mix of assets that best fit your
risk profile - of which long term bonds are a
completely legitimate option, but itś hard to
say without knowing more about your long term
aims/liabilities/job market etc.
DOCUMENT 4 "First off, you generally want
to park your emergency fund somewhere that
is ""safe"", meaning something that is
not subject to market fluctuations. Your
emergency fund is something you need to
be able to count on when times are tough!
That rules out things like stock market
investments. Secondly, you need to think
about how quickly you will need access to
the money. If you have an emergency, odds
are you don´ t want to be waiting around for
weeks/months/years for the money to become
available. This rules out most fixed-term
investments (Bonds, traditional CDs, etc).
If you are concerned that you will need
near-instant access to your emergency money,
then you probably want to keep it in a Savings
or Money Market Account at the same bank as
your checking account. Most banks will let
you transfer money between local accounts
instantly. Unfortunately, your local bank
probably has pitiful interest rates for the
Savings/MMA, far below the inflation rate.
This means your money will slowly lose value
over time. Be prepared to keep contributing
to it! For most people, being able to draw
the cash from your fund within a few days
(<1 week) is sufficient. Worst case, you
charge something on your credit card, and
then pay down the card when the emergency
fund withdrawal arrives. If ""money within
a few days"" is okay for you, there are a
few options: Money Market (Mutual) Funds (not
to be confused with a Money Market Account)
- This is the traditional place to keep an
emergency fund. These are investment funds
you can buy with a brokerage account. An
example of such a fund would be Fidelity
Cash Reserves. MMFs are not FDIC insured,
so they are not exactly zero risk. However,
they are considered extremely safe. They
almost never go down in value (only a few
times in the past few decades), and when
they have, the fund manager or the Federal
Govt stepped in to restore the value. They
usually offer slightly better return than
a local savings account, and are available
in taxable and notaxable varieties. Online
High-Yield Savings or Money Market Account
- These are a relatively new invention. Itś
basically a the same thing as what your
local bank offers, but itś online-only. No
local branch means low overhead, so they
offer higher interest rates (2.0% vs 0.5%
for your local bank). Some of them used
to be over 5% before the economy tanked.
Like your local bank, it is FDIC insured.
One bit of caution: Some of these accounts
have become ""gimmicky"" lately. They have
started to do things like promo rates for a
few months, only offering the high interest
rate on the first few $K deposited, limiting
the amount that can be withdrawn, etc. Be
sure to read the details before you open an
account! No-Penalty CDs - Certificates of
Deposit usually offer a better rate than a
Savings Account, but your money is locked up
until the CD term is up (e.g. 36 months). If
you need to cash out before then, you pay a
penalty. Some banks have begun to offer CDs
that you can cash out with no penalty at all.
These can offer better rates than the savings
account. Make sure it really is no-penalty
though. Also watch what your options are for
slowly adding money over time. This can be
an issue if you want to deposit $100 from
every paycheck. Rewards Checking Accounts -
These are checking accounts that will pay a
relatively high interest rate (3% or more)
provided you generate enough activity. Most
of them will have requirements like you must
have direct deposit setup with them, and
you must do a minimum number of debit card
transactions from the account per month. If
you can stay on top of the requirements, these
can be a great deal. If you don´
t stay on top
of it, your interest rate usually drops back
to something pitiful, though. Personally, we
use the Online High-Yield Savings Account
for our emergency fund. I´ m not going to make
a specific recommendation as to which bank
to use. The best deal changes almost week
to week. Instead, I will say to check out
Bankrate.com for a list of savings accounts
and CDs that you can sort. The Bank Deals
blog is a good place to follow rate changes."
DOCUMENT 5 For maximum liquidity of an
emergency fund, having demand deposits and
fixed deposits would be the best, as they are
the most liquid instruments around. If you
invest in other higher yielding instruments
they would be less liquid, so itś a tradeoff
and you´ ve got to decide for yourself whatś
best given your requirements.n´
,
AGENT REASONING:
The retrieved documents provide multiple
options for storing an emergency fund,
emphasizing liquidity, safety, and moderate
growth. The answer will summarize these
options and their trade-offs.
AGENT RESPONSE:
For your emergency fund, prioritize safety
and liquidity. Here are the best options
based on expert advice:
1. **High-Yield Savings Accounts (HYSA)**:
Offer FDIC insurance, instant access, and
better interest rates than traditional banks
(e.g., online banks like Ally or Marcus).
2. **Money Market Accounts (MMA)**: Similar
to HYSAs but may offer check-writing features;
also FDIC-insured.
3. **Money Market Mutual Funds (MMMF)**:
Slightly higher returns than HYSAs but not
FDIC-insured (though backed by the government
in crises).
4. **No-Penalty CDs**: Provide higher
interest rates than savings accounts without
locking up funds permanently.
5. **Rewards Checking Accounts**: Offer
competitive rates if you meet activity
requirements. Avoid volatile investments like
stocks or bonds, as they risk losing value
during emergencies. Most experts recommend
keeping 3–6 months of expenses in this fund,
prioritizing accessibility over growth.’)
For your emergency fund, prioritize
**safety**, **liquidity**, and **moderate
growth**. Here are the top options based on
expert advice:
1. **High-Yield Savings Accounts (HYSA)**:
- **Pros**: FDIC-insured, instant access,
higher interest rates than traditional banks
(e.g., online banks like Ally, Marcus, or
Capital One). - **Cons**: Returns may lag
inflation.
2. **Money Market Accounts (MMA)**: -
**Pros**: FDIC-insured, offers check-writing
or debit card access, slightly better rates
than HYSAs. - **Cons**: May require minimum
balances.
3. **Money Market Mutual Funds (MMMF)**: -
**Pros**: Slightly higher returns than HYSAs,
highly liquid. - **Cons**: Not FDIC-insured
(but government-backed in crises, e.g., SIPC
protection).
4. **No-Penalty CDs**:
- **Pros**: Higher interest rates than savings
accounts, no early withdrawal penalties.
- **Cons**: Rates may reset or change over
time.
5. **Rewards Checking Accounts**:
- **Pros**: Competitive rates if you meet
activity requirements (e.g., direct deposit,
monthly transactions).
- **Cons**: Rates drop if requirements aren’t
met.
*Avoid**: Stocks, bonds, or other volatile
investments, as they risk losing value when
you need liquidity.
Most experts recommend keeping **3–6 months
of expenses** in this fund. For the best
rates, compare options on platforms like
**Bankrate.com** or track promotions from
online banks. Prioritize accessibility over
high returns, as your emergency fund’s
query rewritten by agent
purpose is to provide a secure safety net.
made of vs made up
of
Explain the grammatical difference between
’made of’ and ’made up of’ with examples
What does "Google-
fu" mean?
C.2 Agent with re-performed retrieval
Here is an example of a flow in which the Agent
did retreival twice:
What does ’Google-fu’ mean and where did it
originate from? Find blog posts explaining this
internet slang term.
How do I invest in
emerging markets
USER QUERY:What’s the difference between
"these" and "those"?
Investing in emerging markets involves under-
standing high-growth potential regions while man-
aging risks like political instability, currency fluc-
tuations, and market volatility. Key strategies in-
clude diversification through ETFs, mutual funds,
or direct stock investments in companies operat-
ing in these regions. Investors should also con-
sider regional economic trends, regulatory envi-
ronments, and long-term growth prospects.
AGENT REASONING:
The user is asking about the difference between
´ theseánd ´ those´ , which are demonstrative
pronouns. To provide an accurate answer,
I need to retrieve relevant blog posts
explaining their usage. Since no previous
results exist, initiating a document retrieval
with a focused query is appropriate
QUERY SENT FOR RETRIEVAL:
difference between these and those in English
grammar
RETRIEVED DOCUMENTS:
DOCUMENT 1 What’s the difference between
these two phrases?
DOCUMENT 2 Is there any difference in meaning
betweeen these two? Is one a valid grammar
construct and the other not?
DOCUMENT 3 Are there any differences between
these two expressions?
DOCUMENT 4 > I am unclear of the use of
[these|those] objects. I am unsure when to
use > [these|those|them]. Please someone help
me tell me which is correct in the previous
sentences. This is not a dupe of What
ś the difference between "these" and "those"?
DOCUMENT 5 What is the difference? Or is there
any? Which would be more British English?
AGENT REASONING:
The initial retrieved documents do not provide
a clear explanation of the difference between
´ theseánd ´ those´ . A more specific search is
needed to find blog posts that directly
address their correct usage and distinctions
in English grammar.
QUERY SENT FOR RETRIEVAL:
correct usage and difference between these
and those in English grammar
RETRIEVED DOCUMENTS:
DOCUMENT 1 > I am unclear of the use of
[these|those] objects. I am unsure when to
use > [these|those|them]. Please someone help
me tell me which is correct in the previous
sentences. This is not a dupe of What
ś the difference between "these" and "those"?
DOCUMENT 2 Which is the correct use of these
two words, and in which context should one be
used rather than the other?
DOCUMENT 3 Is there a difference between
these two expressions and should one of them
be preferred?
DOCUMENT 4 What is the difference between
those questions and which one is the correct
Does the currency
exchange rate con-
tain any additional
information at all?
What factors beyond basic numerical values are
embedded in currency exchange rates? How
do economic indicators, market conditions, and
geopolitical events influence or get reflected in
exchange rates? Are there additional layers of in-
formation (e.g., forward rates, volatility measures)
associated with currency exchange rates?
Table 10: Examples of query rewriting performed by
Agentic and Enhanced RAG
form?
DOCUMENT 5 Which is the correct grammar?
Which is correct?
AGENT RESPONSE:
Here are some blog posts that might be
relevant to your query:
"These vs. Those: Clarifying Usage" –
Explores when to use "these" and "those" in
sentences. *(Document 4, Step 0)*
"Correct Grammar: These or Those?" –
Discusses valid grammar constructs and
contextual preferences. *(Document 2, Step
1)*
"Differences in Expressions" – Analyzes
variations between phrases using "these" and
"those." *(Document 3, Step 0)*
"Choosing Between These and Those" – Addresses
confusion in selecting the right pronoun for
objects. *(Document 1, Step 1)*
"British vs. American English Usage" –
Compares regional preferences for "these" and
"those." *(Document 5, Step 0)*
C.3 Agent rewriting
In Table 10 we report a few examples on how query
rewriting is performed by the Agent.
Figure 5: Overall computational cost and token usage for each model in the Agentic setting when processing a
single user query. Qwen3 4B, 8B, and 32B operate in thinking mode, whereas the 0.6B variant is run without it.
Qwen3 0.6B, 4B, and 8B are executed on a single NVIDIA A40, while Qwen3 32B is run on 4×A40 GPUs.
Top: Distribution of total latency, measured from the moment the system receives the query to the moment the final
answer is produced. Within the Qwen family, Qwen3-0.6B achieves the lowest latency due to its smaller size and
the absence of thinking mode. Middle: Average number of input tokens. This value increases slightly with model
size. The peak of low input-token values for GPT-4.1-nano arises because the model frequently opts not to use the
RAG tool, thereby reducing the number of required reasoning steps.
Bottom: Average number of output tokens. Here, the largest difference emerges: enabling thinking mode leads
Qwen3 4B, 8B, and 32B to produce substantially longer outputs.