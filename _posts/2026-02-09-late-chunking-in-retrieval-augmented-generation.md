---
layout: post
title: "The Paradigm Shift to Late Chunking in Retrieval-Augmented Generation: A Technical and Comparative Analysis"
---

## **Executive Summary**

The architecture of Retrieval-Augmented Generation (RAG) systems is currently undergoing a significant transformation, driven by the persistent inability of traditional retrieval pipelines to maintain semantic coherence across fragmented document segments. For years, the industry standard has relied on "early chunking" strategies—whether naive, fixed-size splitting or more sophisticated semantic segmentation—prior to vector embedding. While computationally expedient, these methods fundamentally sever the semantic connective tissue of long-form text, isolating pronouns from their referents and stripping local segments of their governing global context. This report provides an exhaustive technical analysis of "Late Chunking," a methodology championed by Jina AI and Voyage AI that reverses the traditional pipeline order to preserving the full semantic context of documents within individual chunk embeddings.

This document serves as a definitive guide for engineering teams currently utilizing semantic and LLM-based chunking strategies. We deconstruct the mechanical underpinnings of long-context transformers that enable late chunking, contrasting Jina AI’s "mean pooling over token maps" approach with Voyage AI’s "contextualized chunk embeddings." Through a rigorous comparative analysis, we evaluate these methods against existing pipelines, highlighting critical trade-offs regarding indexing latency, storage architecture, and the phenomenon of "context bleeding." The analysis indicates that while late chunking introduces non-trivial indexing overhead and engineering complexity regarding token alignment, it solves the "Lost in the Middle" pathology and anaphora resolution problems inherent in traditional RAG, offering a superior balance of retrieval accuracy and query-time latency compared to both naive methods and Late Interaction (ColBERT) architectures.

## **1\. The Context Dilemma in Retrieval-Augmented Generation**

To understand the necessity of late chunking, one must first deconstruct the limitations of the prevailing architectures governing current RAG systems. The fundamental promise of RAG is to ground Large Language Model (LLM) generation in verified, external knowledge. However, the retrieval step acts as a bottleneck; if the semantic signal of a document is fractured during the indexing phase, the retrieval system fails to surface the relevant context, regardless of the subsequent LLM's reasoning capabilities.

### **1.1 The Pathology of Naive and Early Chunking**

The standard preprocessing pipeline for RAG systems typically involves a process known as "early chunking." In this paradigm, a lengthy document—such as a 10,000-token financial report or a complex legal agreement—is segmented into smaller, discrete units before any neural processing occurs. These segments, often sized between 256 and 1024 tokens, are treated as independent data points. The motivation for this approach is largely pragmatic: standard embedding models (like the original BERT or early OpenAI models) had strict input limits, typically 512 or 8192 tokens, and processing smaller chunks allowed for efficient batching and storage.

However, the pathology of this approach lies in the semantic independence enforced upon the chunks. Once a document is sliced, the embedding model treats each segment as a distinct universe, devoid of any knowledge of the preceding or succeeding text. This leads to two primary failures: anaphora severance and thematic drift.

Anaphora severance occurs when a pronoun or reference in one chunk refers to an entity defined in a previous chunk. For instance, if Chunk 1 introduces "Project Apollo" and Chunk 2 discusses "its budget overrun," the embedding for Chunk 2 encodes the vector for "its budget" without any mathematical relation to "Project Apollo." In the high-dimensional vector space, the representation for Chunk 2 floats in a cluster of generic "budget" concepts, potentially near financial markets or household spending, far from the specific entity "Project Apollo".<sub>1</sub> Consequently, a user query specifically asking about "Project Apollo's budget" will fail to retrieve Chunk 2 because the vector similarity search finds no overlap between "Project Apollo" and the generic "its budget."

Thematic drift is a more subtle but equally damaging issue, particularly in technical and legal documentation. Prerequisites, definitions, or safety warnings defined in the preamble of a document often govern the interpretation of subsequent procedures. Naive chunking divorces the procedure from its governing context. A chunk describing a chemical mixing process might be retrieved in isolation, missing the crucial safety warning located in the document's introduction, leading to the retrieval of dangerous or incorrect instructions that lack their safety context. The embedding model, seeing only the procedure, encodes it as a standard instruction set, indistinguishable from similar instructions in a different, perhaps benign, context.

### **1.2 The Limitations of Semantic Chunking**

Engineering teams, including your own, often graduate from naive chunking to "semantic chunking" to mitigate these coherence issues. Semantic chunking employs a sliding window approach to compare the cosine similarity or other distance metrics of adjacent sentences or propositions. When the similarity drops below a predefined threshold, a breakpoint is introduced. The theory is that this drop in similarity represents a topic shift, and thus chunks are created based on semantic boundaries rather than arbitrary character counts.

While semantic chunking represents an improvement over fixed-size splitting—primarily by ensuring that paragraphs are not split in the middle of a sentence and that distinct topics are kept somewhat separate—it remains a "pre-embedding" strategy.<sub>3</sub> It attempts to guess boundaries based on local coherence but fails to capture global dependencies. It does not solve the long-range dependency problem. A semantic chunker might successfully group a paragraph about "engine maintenance" together, but if that paragraph relies on a definition of "engine" provided twenty pages earlier, the semantic chunking algorithm has no mechanism to inject that definition into the maintenance chunk. The resulting embedding is still locally coherent but globally isolated. The semantic chunker is effective at identifying *where* a topic changes, but it does nothing to preserve the *content* of the previous topic within the current chunk's representation.<sub>5</sub>

### **1.3 The Computational Cost of LLM-Based Contextual Retrieval**

A more recent innovation, often referred to as "Contextual Retrieval" (popularized by Anthropic and utilized in your current stack), employs an LLM to preprocess chunks. In this workflow, the system iterates through every chunk of the document. For each chunk, it passes both the chunk and the full document (or a large window of it) to an LLM with a prompt to "situate this chunk within the context of the document." The LLM generates a summary or a clarifying preamble, which is then explicitly prepended to the chunk text before embedding.

* **Mechanism:** Chunk\_New \= "Document Title: X. Summary: Y. " \+ Chunk\_Original

This approach explicitly re-injects lost context, solving the anaphora problem by forcing the LLM to rewrite "its budget" as "Project Apollo's budget." However, this method introduces significant operational drawbacks. It is prohibitively expensive and slow at scale. Indexing a massive corpus requires an LLM inference pass for every single chunk, increasing indexing costs by orders of magnitude compared to simple BERT-based embedding.<sub>6</sub> Furthermore, it introduces non-determinism; the LLM might hallucinate context, frame it inconsistently, or focus on irrelevant details, adding noise to the vector representation. The "context" is added as text, which increases the token count and can dilute the original semantic signal of the chunk itself.

### **1.4 The Emergence of Late Chunking**

Late Chunking proposes a radical restructuring of the pipeline to address these limitations without the prohibitive cost of LLM generation: **Embed first, chunk later.**

By utilizing the massive context windows of modern BERT-derived transformers (such as Jina's 8192-token window), the system ingests the *entire* document (or very large sections of it) in a single pass. The transformer’s self-attention mechanism computes the relationship between every token and every other token in the document.<sub>1</sub>

Only *after* the model has computed these context-aware token representations—where the vector for "it" in paragraph 5 has mathematically attended to "Project Apollo" in paragraph 1—does the system slice the vectors into chunks. This preserves the global context within the local geometry of the chunk's embedding. This shift moves the chunking operation from the pre-processing stage to the post-processing stage of the embedding model, fundamentally altering how information is encoded and retrieved.

## **2\. Theoretical Underpinnings of Late Chunking**

To fully appreciate the efficacy of late chunking, one must delve into the mathematical and architectural mechanisms of the transformer models that power it. The core innovation relies on the specific behavior of the self-attention mechanism and how it differs from independent processing.

### **2.1 The Transformer Attention Field**

In a standard encoder-only transformer (like BERT), the input is a sequence of tokens ![][image1]. The self-attention mechanism generates a new representation for each token, ![][image2], which is a weighted sum of all other token representations in the sequence.

![][image3]  
This mechanism allows every token to "attend" to every other token, gathering information relevant to its own interpretation. The attention weights determine how much influence token ![][image4] has on token ![][image5]. In a naive approach, if we split the document at token ![][image6], the representation for token ![][image7] is calculated as ![][image8]. The model is mathematically blind to tokens ![][image9]. The attention mask prevents any information flow from the first part of the document to the second.

In **Late Chunking**, the model processes the full sequence ![][image10] (where ![][image11] can be 8192 tokens or more). The representation for token ![][image7] becomes ![][image12]. Crucially, the vector ![][image13] now mathematically encodes information from ![][image14], even though ![][image14] will eventually belong to a different chunk. The embedding for the word "bank" in the sentence "He sat on the bank" will be fundamentally different if the preceding 5000 tokens discuss river ecosystems versus financial institutions. Late chunking ensures this disambiguation happens *before* the document is segmented.<sub>2</sub>

### **2.2 The "Bleeding" of Context**

This phenomenon can be described as "context bleeding" or "semantic diffusion." In a high-dimensional semantic space, the vector for a generic sentence is shifted by its context.

* **Naive Chunk Vector:** The vector for the sentence "It declined by 5%" sits in a cluster of generic "decline" vectors, essentially a centroid of the words "declined" and "5%". It is equidistant to documents about stock markets, battery levels, and population statistics.  
* **Late Chunk Vector:** The vector for "It declined by 5%"—when processed with a document about "Tesla Stock"—incorporates the "Tesla" attention signal. The token embedding for "It" is shifted toward the concept of "Tesla" and "Stock". When these token embeddings are pooled (averaged) to form the chunk vector, the resulting vector moves from the generic "decline" cluster to the specific "Tesla financial decline" cluster.

This "bleeding" ensures that the vector is discriminative against similar sentences from different documents (e.g., "It declined by 5%" in a document about rainfall). The mathematical representation of the chunk is no longer just the sum of its parts; it is the sum of its parts conditioned on the whole.<sub>10</sub>

## **3\. Architectural Deep Dive: Jina AI and Voyage AI**

Two primary vendors have pioneered the implementation of late chunking: Jina AI and Voyage AI. While both aim to solve the context loss problem, their architectural approaches and implementation details differ significantly.

### **3.1 Jina AI: Mean Pooling Over Token Maps**

Jina AI’s implementation of late chunking relies on their open-weights models, specifically jina-embeddings-v2, which utilize an architecture modified to support long contexts efficiently.

**The ALiBi Mechanism:** Jina's models replace standard positional embeddings with ALiBi (Attention with Linear Biases). Standard Transformers use absolute positional embeddings, which struggle to generalize beyond the sequence length seen during training. ALiBi encodes position by biasing the query-key attention scores based on their distance. This allows the model to extrapolate to longer sequences (up to 8192 tokens) without the quadratic performance degradation associated with some other long-context techniques.<sub>1</sub>

**The Late Chunking Algorithm:**

1. **Full Document Tokenization:** The entire document is tokenized into a single sequence input\_ids. If the document exceeds 8192 tokens, a sliding window strategy (discussed later) is employed.  
2. **Transformer Inference:** The model processes the input\_ids and outputs a tensor of shape (Batch\_Size, Sequence\_Length, Hidden\_Dimension). For jina-embeddings-v2-base-en, this is (1, 8192, 768). This tensor contains the *contextualized* embedding for every token.  
3. **Boundary Alignment:** The developer must provide a map of chunk boundaries. This is typically generated *a priori* using a sentence splitter or semantic segmenter. For example, Chunk 1 corresponds to tokens 0-50, Chunk 2 to tokens 51-120.  
4. **Span Pooling:** Instead of pooling the entire tensor into a single document vector, the system performs mean pooling on the specific slices of the tensor corresponding to the chunk boundaries.  
   * Vector\_Chunk\_1 \= Mean(Tensor\[0, 0:50, :\])  
   * Vector\_Chunk\_2 \= Mean(Tensor\[0, 51:120, :\])

This approach is "passive" in that the model is a standard embedding model; the "late chunking" is a procedural operation performed on the output hidden states.<sub>14</sub>

### **3.2 Voyage AI: Contextualized Chunk Embeddings**

Voyage AI introduces a variation termed "Contextualized Chunk Embeddings," specifically with their voyage-context-3 model. While Jina's approach is a post-processing technique on a general model, Voyage's approach appears to be an algorithmic feature baked into their API and training objective.<sub>16</sub>

**Mechanism Differences:**

Voyage AI critiques Jina's late chunking as offering only "Partial" context preservation. They argue that simply averaging the token embeddings—even if contextualized—may not optimally represent the chunk for retrieval tasks. Voyage's model is trained with a specific objective to optimize the chunk embedding using the global context.

* **Instruction Tuning:** Voyage's API typically requires specifying an input\_type (e.g., "document") and potentially allows for instruction-based tuning where the model is prompted to "embed this chunk given this context."  
* **API Abstraction:** Unlike Jina, where the developer must handle the tensor slicing and token mapping (often using the pylate library), Voyage wraps this complexity. The developer sends the document and the chunk boundaries (or the chunks themselves), and the API returns the optimized vectors.  
* **Performance Claims:** Voyage AI claims significant performance leads over Jina-v3 late chunking, citing approximately 23% better performance on chunk-level retrieval benchmarks. They attribute this to a "Full, Principled" context preservation strategy that goes beyond simple attention side-effects.<sub>16</sub>

**Closed vs. Open:**

Crucially, Jina's approach uses open-weights models that can be self-hosted and inspected. Voyage AI operates a closed API. This has significant implications for data privacy, vendor lock-in, and cost control, which will be discussed in the comparative analysis.

## **4\. Comparative Analysis: Late Chunking vs. Incumbent Strategies**

To evaluate the strategic value of adopting late chunking, we must benchmark it directly against the methods currently employed in your stack: Semantic Chunking and LLM-Based Contextual Retrieval.

### **4.1 Late Chunking vs. Semantic Chunking**

**Current State Analysis:**

Your current semantic chunking pipeline likely uses a sliding window to calculate embedding similarities between sentences. When the similarity dips, a new chunk is started. This effectively groups sentences that are *locally* related.

| Feature | Semantic Chunking (Current) | Late Chunking (Proposed) |
| :---- | :---- | :---- |
| **Context Scope** | Local (Sentence N vs N+1) | Global (Sentence N vs Document) |
| **Dependency Resolution** | Fails for long-range refs (e.g., pronouns referring to pg 1\) | Excellent (Attention mechanism resolves refs) |
| **Coherence** | High (groups related sentences) | High (can use semantic boundaries for slicing) |
| **Indexing Speed** | Fast (Small window processing) | Moderate (Requires full document attention) |
| **Storage Footprint** | Standard | Standard |

**The Delta:**

The primary deficit of semantic chunking is that while it respects *topic* boundaries, it does not preserve *information* across them. If a semantic chunker correctly identifies a paragraph about "Billing" as distinct from "Shipping," it separates them. However, if the "Billing" section relies on a payment term defined in the "Introduction," semantic chunking severs that link. Late chunking can utilize the exact same boundaries identified by your semantic chunker but ensures that the vector for the "Billing" chunk mathematically contains the "Introduction" context.

**Strategic Insight:** Late chunking should be viewed not as a *replacement* for semantic segmentation logic, but as a replacement for the *embedding* step that follows it. You can continue to use your semantic chunker to define *where* to cut, but use late chunking to determine *what* the cut contains.<sub>3</sub>

### **4.2 Late Chunking vs. LLM-Based Contextual Retrieval**

**Current State Analysis:**

Your use of LLM-based chunking (generating summaries/context for each chunk) is the current gold standard for quality but the bottleneck for cost and latency.

| Feature | LLM-Based Contextual Retrieval (Current) | Late Chunking (Proposed) |
| :---- | :---- | :---- |
| **Context Mechanism** | Explicit Text ("This chunk is about...") | Implicit Vector Weights (Attention) |
| **Indexing Cost** | (LLM Inference per chunk) | $$ (Embedding Inference per doc) |
| **Indexing Latency** | Very High (Sequential generation) | Moderate (Parallelizable) |
| **Determinism** | Low (LLM hallucination/variance) | High (Mathematical operation) |
| **Searchability** | Modified by added text (Hybrid search capable) | Pure Vector (Dense retrieval only) |
| **Context Bleeding** | Controlled by prompt | High (Automatic attention) |

**The Delta:** LLM-based retrieval is "brute force" context injection. It works exceptionally well because it converts implicit context into explicit keywords. However, it is operationally heavy. Late chunking offers a "pareto-optimal" alternative: it achieves 80-90% of the retrieval quality of LLM-augmentation at 10% of the cost and 100x the speed.<sub>2</sub>

**Blind Spot \- "Context Bleeding":** One area where LLM-based chunking may outperform late chunking is in *negative* discrimination. An LLM can be prompted to "summarize only the relevant context." Late chunking's attention mechanism is indiscriminate; it attends to everything. If a document contains two contradictory sections (e.g., "Plan A" and "Plan B"), the embedding for a chunk in "Plan A" might absorb tokens from "Plan B" simply because they are in the same document. This "bleeding" can arguably dilute the specific signal of the chunk, causing a query for "Plan B" to retrieve "Plan A" because they share a "document-level" vector signature. LLM augmentation avoids this by explicitly stating "This chunk describes Plan A.".<sub>2</sub>

### **4.3 Late Chunking vs. Late Interaction (ColBERT)**

While not part of your current stack, Late Interaction (ColBERT) is the main alternative for high-performance retrieval.

| Feature | Late Interaction (ColBERT) | Late Chunking |
| :---- | :---- | :---- |
| **Storage Mechanism** | Vector per Token (Massive Index) | Vector per Chunk (Standard Index) |
| **Storage Cost** | \~2.5 TB per 100k docs | \~5 GB per 100k docs |
| **Retrieval Latency** | High (Complex scoring) | Low (Standard ANN search) |
| **Compatibility** | Requires specialized engine (Plaidx, etc.) | Compatible with any Vector DB |

**The Delta:** ColBERT stores a vector for every token, deferring the interaction until query time (hence "Late Interaction"). Late Chunking bakes the interaction into the chunk vector at indexing time. Late chunking effectively compresses the benefits of ColBERT into a standard vector size, offering a massive reduction in storage requirements (approx. 500x reduction) while maintaining compatibility with standard vector databases like Pinecone, Milvus, or Weaviate.<sub>20</sub>

## **5\. Performance Evaluation and Benchmarks**

The empirical evidence supporting late chunking is robust, particularly for long-context retrieval tasks.

### **5.1 Retrieval Metrics (nDCG@10)**

Benchmarks cited in Jina AI's research and independent replications demonstrate consistent improvements over naive chunking 1:

* **SciFact**: \+1.9% improvement (From 64.20% to 66.10%).  
* **NFCorpus**: \+6.5% improvement (From 23.46% to 29.98%). This is a significant jump, indicating that for medical/scientific texts where context is dense and distributed, late chunking is superior.  
* **FiQA2018**: \+0.6% improvement.  
* **Berlin Wikipedia Test**: Improved cosine similarity for anaphora resolution from \~0.75 to \~0.85.

### **5.2 Recall vs. Precision**

Late chunking generally improves **Recall** significantly by ensuring that chunks with ambiguous phrasing ("it", "the project") are retrieved when the query includes the specific entity name. However, as noted in the "Context Bleeding" section, there is a theoretical risk to **Precision** if the document-level context overwhelms the specific chunk detail, though benchmarks suggest the trade-off is net positive.

### **5.3 Voyage AI Benchmarks**

Voyage AI reports even higher gains for their specific implementation 16:

* **Chunk-Level Retrieval**: Voyage-context-3 outperforms Jina-v3 late chunking by \~23%.  
* **Document-Level Retrieval**: Outperforms Jina by \~20%.  
* **Comparison to Contextual Retrieval (LLM)**: Voyage claims to match or exceed the performance of LLM-augmented retrieval while being faster and cheaper.

## **6\. Implementation Realities and Engineering Challenges**

Transitioning to late chunking is not merely a model swap; it requires a fundamental re-engineering of the indexing pipeline. This section details the "blind spots" and practical challenges your developers will face.

### **6.1 The "O(n²)" Compute Spike (Indexing Latency)**

**The Blind Spot:** Developers often underestimate the computational cost of processing 8192 tokens in a single transformer pass compared to processing smaller chunks.

**The Reality:** Transformer attention scales quadratically with sequence length (![][image15]). While Jina v2 uses ALiBi and Flash Attention to mitigate this, embedding a single 8000-token document is significantly more computationally intensive than embedding sixteen 500-token chunks independently.

* **Throughput Impact:** Your ingestion throughput (docs/second) will drop significantly. You cannot simply blast the model with batches of 32 documents if they are all 8k tokens length; you will run out of GPU memory (OOM).  
* **Batching Strategy:** You must implement dynamic batching or set batch size to 1 for long documents on standard consumer GPUs (e.g., A10G, T4).

### **6.2 The Token Alignment Nightmare**

**The Blind Spot:** Mapping vector slices back to human-readable text.

**The Reality:** Late chunking operates on *tokens*. RAG systems display *text*.

* **The Issue:** Tokenizers are destructive. They handle whitespace, special characters, and unknown tokens in ways that do not map 1:1 with character indices. If your vector slice \[20:40\] corresponds to tokens, but you blindly slice the string using a character estimation, you will display broken text (e.g., "pple" instead of "Apple").  
* **Requirement:** You must maintain a strict Token\_ID \-\> (Start\_Char, End\_Char) map during the ingestion process.  
* **Tooling:** Jina's ecosystem provides tools like pylate to handle this, but if you are building custom logic (e.g., in LangChain), you must implement this mapping carefully. A misalignment of one token can shift the semantic meaning of the retrieved chunk.<sub>11</sub>

### **6.3 Handling Documents \> 8192 Tokens**

**The Blind Spot:** Documents that exceed the model's maximum context window.

**The Reality:** A 50-page legal document (approx. 25k tokens) cannot be processed in a single pass even by Jina v2.

* **Sliding Window Late Chunking:** You must implement a sliding window strategy (e.g., Window 1: 0-8192, Window 2: 4096-12288).  
* **The Overlap Problem:** A chunk located in the overlap region (e.g., at token 6000\) will receive *two* different embeddings: one from Window 1 and one from Window 2\.  
* **Resolution Strategy:** You must implement logic to either average these embeddings or select the one where the chunk is most centrally located (to maximize bidirectional context). This adds significant complexity to the ingestion logic.<sub>22</sub>

### **6.4 Vendor Lock-In and Model Availability**

**The Blind Spot:** Dependency on specific model architectures.

**The Reality:** Late chunking requires access to the *hidden states* of the transformer.

* **OpenAI Incompatibility:** You **cannot** implement late chunking with text-embedding-3-small or large because OpenAI's API only returns the final pooled vector. It does not expose the token-level hidden states required for the late pooling operation.  
* **Implication:** You must commit to using either open-weights models (Jina, Nomic, GTE) which you host yourself (or via HuggingFace Inference Endpoints), or use specific APIs like Voyage that explicitly support this feature. This reduces your ability to easily swap embedding providers in the future.<sub>23</sub>

## **7\. Strategic Blindspots and Operational Risks**

Beyond the engineering challenges, there are broader operational risks to consider.

### **7.1 Re-Indexing Costs and Flexibility**

In a naive chunking pipeline, if you decide to change your chunk size from 512 to 256 tokens, you re-split the text and re-embed. In late chunking, if you store the *full token tensor* (8192 x 768 floats), you could theoretically re-pool without re-inference. However, storing these tensors is prohibitively expensive (approx. 25MB per document).

**The Reality:** You will likely perform the pooling immediately and discard the tensor. Therefore, if you want to change your chunk boundaries later (e.g., switch from paragraph-based to sentence-based), you must re-run the expensive 8k-token inference for your entire corpus. The "flexibility" of late chunking is theoretical unless you have massive storage capacity.

### **7.2 Context Dilution**

While "context bleeding" helps disambiguate chunks, it can also homogenize them. In a document that covers a diverse range of topics, the global attention mechanism might cause distinct topics to "pollute" each other. **Mitigation:** For extremely long and diverse documents (e.g., a "History of the World"), it might be beneficial to segment the document *before* late chunking into large, coherent chapters (e.g., "The Roman Empire") and apply late chunking only within those chapters. This prevents the "Industrial Revolution" context from bleeding into the "Roman Empire" chunks.<sub>19</sub>

## **8\. Conclusion and Strategic Recommendations**

Late Chunking represents a significant advancement in neural information retrieval, effectively addressing the "Lost in the Middle" phenomenon without the prohibitive costs of LLM-based augmentation or the storage demands of ColBERT. For a developer working on RAG systems, it offers a mathematically sound approach to resolving the context-granularity trade-off.

**Strategic Recommendations:**

1. **Replace LLM-based Chunking:** If cost and indexing latency are concerns, Late Chunking is a viable replacement for LLM-based Contextual Retrieval. It delivers comparable context preservation for dense technical documents at a fraction of the cost.  
2. **Augment, Don't Replace, Semantic Logic:** Use your existing semantic chunking algorithms to define the *boundaries* (the start and end indices) of your chunks. Pass these boundaries to the late chunking embedding model. This combination—semantic boundaries \+ late contextualized embedding—represents the current state-of-the-art for dense retrieval.  
3. **Invest in Infrastructure:** Prepare your ingestion pipeline for GPU-heavy inference. Ensure your vector database and retrieval logic are decoupled from the embedding generation to allow for the specific batching requirements of long-context models.  
4. **Evaluate Voyage vs. Jina:**  
   * Choose **Voyage AI** if you prefer a managed API, require the highest possible performance (according to their benchmarks), and are comfortable with a closed ecosystem.  
   * Choose **Jina AI** if you require on-premise deployment, data sovereignty, or lower costs via self-hosting open-weights models.

By shifting the chunking operation to the post-embedding stage, you effectively allow the system to "read the whole book" before asking it to "summarize the page," resulting in vectors that are significantly more aligned with the semantic intent of complex user queries.

#### **Works cited**

1. Late Chunking in Long-Context Embedding Models \- Jina AI, accessed on February 6, 2026, [https://jina.ai/news/late-chunking-in-long-context-embedding-models/](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)  
2. Late Chunking vs Contextual Retrieval: The Math Behind RAG's Context Problem \- Medium, accessed on February 6, 2026, [https://medium.com/kx-systems/late-chunking-vs-contextual-retrieval-the-math-behind-rags-context-problem-d5a26b9bbd38](https://medium.com/kx-systems/late-chunking-vs-contextual-retrieval-the-math-behind-rags-context-problem-d5a26b9bbd38)  
3. arXiv:2504.19754v1 \[cs.IR\] 28 Apr 2025, accessed on February 6, 2026, [https://arxiv.org/pdf/2504.19754](https://arxiv.org/pdf/2504.19754)  
4. 8 Types of Chunking for RAG Systems \- Analytics Vidhya, accessed on February 6, 2026, [https://www.analyticsvidhya.com/blog/2025/02/types-of-chunking-for-rag-systems/](https://www.analyticsvidhya.com/blog/2025/02/types-of-chunking-for-rag-systems/)  
5. Chunking Strategies for Retrieval-Augmented Generation (RAG): A Deep Dive into SemDB's Approach \- Intelligence Factory AI, accessed on February 6, 2026, [https://www.intelligencefactory.ai/blog/chunking-strategies-for-retrieval-augmented-generation-rag-a-deep-dive-into-semdbs-approach](https://www.intelligencefactory.ai/blog/chunking-strategies-for-retrieval-augmented-generation-rag-a-deep-dive-into-semdbs-approach)  
6. Chunking Strategies to Improve Your RAG Performance \- Weaviate, accessed on February 6, 2026, [https://weaviate.io/blog/chunking-strategies-for-rag](https://weaviate.io/blog/chunking-strategies-for-rag)  
7. The Evolution of RAG Text Chunking: Why Precision Still Matters - by Tao An \- Medium, accessed on February 6, 2026, [https://tao-hpu.medium.com/the-evolution-of-rag-text-chunking-why-precision-still-matters-c3e35ef79c50](https://tao-hpu.medium.com/the-evolution-of-rag-text-chunking-why-precision-still-matters-c3e35ef79c50)  
8. Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models - alphaXiv, accessed on February 6, 2026, [https://www.alphaxiv.org/overview/2409.04701v2](https://www.alphaxiv.org/overview/2409.04701v2)  
9. Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models \- arXiv, accessed on February 6, 2026, [https://arxiv.org/html/2409.04701v3](https://arxiv.org/html/2409.04701v3)  
10. Stop Losing Context\! How Late Chunking Can Enhance Your Retrieval Systems - by Barhoumi Mosbeh - Towards AI, accessed on February 6, 2026, [https://pub.towardsai.net/late-chunking-in-long-context-embedding-models-caf1c1209042](https://pub.towardsai.net/late-chunking-in-long-context-embedding-models-caf1c1209042)  
11. Traditional vs. Late Chunking. Revolutionizing Context Preservation in… - by Bharatkumar kori - Accredian - Medium, accessed on February 6, 2026, [https://medium.com/accredian/traditional-vs-late-chunking-ed1a70022ff4](https://medium.com/accredian/traditional-vs-late-chunking-ed1a70022ff4)  
12. jina-embeddings-v2-base-en \- Search Foundation Models, accessed on February 6, 2026, [https://jina.ai/models/jina-embeddings-v2-base-en/](https://jina.ai/models/jina-embeddings-v2-base-en/)  
13. Papers Explained 264: Jina Embeddings v2 - by Ritvik Rastogi \- Medium, accessed on February 6, 2026, [https://ritvik19.medium.com/papers-explained-264-jina-embeddings-v2-c5d540a9154f](https://ritvik19.medium.com/papers-explained-264-jina-embeddings-v2-c5d540a9154f)  
14. Late Chunking for RAG: Implementation With Jina AI - DataCamp, accessed on February 6, 2026, [https://www.datacamp.com/tutorial/late-chunking](https://www.datacamp.com/tutorial/late-chunking)  
15. jina-ai/late-chunking: Code for explaining and evaluating late chunking (chunked pooling) \- GitHub, accessed on February 6, 2026, [https://github.com/jina-ai/late-chunking](https://github.com/jina-ai/late-chunking)  
16. Introducing voyage-context-3: focused chunk-level details with ..., accessed on February 6, 2026, [https://blog.voyageai.com/2025/07/23/voyage-context-3/](https://blog.voyageai.com/2025/07/23/voyage-context-3/)  
17. SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic Association and Long Story Comprehension \- arXiv, accessed on February 6, 2026, [https://arxiv.org/html/2508.01959v1](https://arxiv.org/html/2508.01959v1)  
18. Contextualized Chunk Embeddings \- Voyage AI by MongoDB, accessed on February 6, 2026, [https://www.mongodb.com/docs/voyageai/models/contextualized-chunk-embeddings/](https://www.mongodb.com/docs/voyageai/models/contextualized-chunk-embeddings/)  
19. Late chunking in Elasticsearch with Jina Embeddings v2, accessed on February 6, 2026, [https://www.elastic.co/search-labs/blog/late-chunking-elasticsearch-jina-embeddings](https://www.elastic.co/search-labs/blog/late-chunking-elasticsearch-jina-embeddings)  
20. Late Chunking: Balancing Precision and Cost in Long Context Retrieval - Weaviate, accessed on February 6, 2026, [https://weaviate.io/blog/late-chunking](https://weaviate.io/blog/late-chunking)  
21. I tested different chunks sizes and retrievers for RAG and the result surprised me \- Reddit, accessed on February 6, 2026, [https://www.reddit.com/r/Rag/comments/1ov0pzk/i\_tested\_different\_chunks\_sizes\_and\_retrievers/](https://www.reddit.com/r/Rag/comments/1ov0pzk/i_tested_different_chunks_sizes_and_retrievers/)  
22. Late Chunking on long documents \- cck's site, accessed on February 6, 2026, [https://enzokro.dev/blog/blog\_post?fpath=blog%2F008\_long\_late\_chunking%2Flong\_late\_chunking.ipynb](https://enzokro.dev/blog/blog_post?fpath=blog/008_long_late_chunking/long_late_chunking.ipynb)  
23. Choosing the Right Chunking Strategy: A Comprehensive Guide to RAG Optimization, accessed on February 6, 2026, [https://dev.to/vishalmysore/choosing-the-right-chunking-strategy-a-comprehensive-guide-to-rag-optimization-4nan](https://dev.to/vishalmysore/choosing-the-right-chunking-strategy-a-comprehensive-guide-to-rag-optimization-4nan)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJMAAAAYCAYAAAD+ks8OAAAFEElEQVR4Xu2ad6gdRRTGP1vsJWqiWJ8lIhZUEkVFzdqVCDZEsZCIYEEFRSSxgmLviIgFyUMl2FD8xxr1WVBDFAOJihjxYcVICHYwiXo+vh3cnLdz7+7NnfsK+4OPd+85d+fNvTtz5pyZBRoaGhoauscWpt28saHrrGfaxhvHCkeaPjV9Z7rd+Rq6T5/pKdMC0xzT+qt4SzjINL+C3je9ZdpHl/WcrU2/mR4wrel8DWnZ0rTY9LR3eB41fWKaZtoeummvmf41nWSakNun57ZddVnPORr6/8d7R4E+04vQl+82fUjXdmrGmR4zHe4dNbjf9Lc3Flnb9IVpc2f/EYoC9Bfh8rKWs/UKDiIOJi51MW42LTdt6B1dIGXbqQkT8VjvqMG9UBtR+E987rE7dNHLzs6l5SNn6yUnQP3KnL3IB6YPvbFLpGw7NbdCE2Ej76jB3dDvH00xZph2cbYLoYtmOvvGpmucrZecCPVrqrOvAy29e0JhmOF8kmnT4oc6JGXbvWAnqL/zTAuhCphpTCfcCf3+tVYmJlm86ADvGGZOgfp1qLMfYnrVtAjys1Dg+1a5VVVStp2aTaC+vgv1/fP8vQ8SVeEKxnY4wSrzk+lX1ByBOY+Y3o5oAKoGqTdNb0CRsSo3mn6HCoIyUuY0KdtOzTHQIODf1eFUlE/mKAznvOAl7xhGGKavNi0xneN8RRg1mNeUsZXpfG+sQaxt5g/Hma4z7ed8I4Vb0L2JwNzrG6iy90XbEC6GBtOV3jGMsBz/A7phsRDLH4o5Db+shyXtK6bvvaMirdq+FhpMmekrrN6ATUVsInQCt4gGoKr+slVdQ3kOGkz7e0dFJkMVYlVV3bMab/rMNNc7ckIo540tI0PngynW9rrQID85f8+BxejZSXqQijARbvOODmB0/8X0kGkN5xsCP/AzOs+XyAXQj1pVR+iySpwN3dR9vQMK5SugipPMgpbsQIb4YBoHLaUxWrV9OnROSC41Deavi+zhDQ7+b78RyrzQTzT2k5PVszN0hlZGmAihYMhM5+Wvued0g2kidN/uQuu86nqoLSb2bWH1xg+/7h0jhLA1cJh3QOdHi/PXPO55puAjGeKDqd/0j+ksZw+0a5twInIP7nJnnwH1+WFnD3BbhgM1tB/4Gspz+gq2fqgtTqoAS3/2nRvPZYRtnu2gycDCh4OBUZXbPDya4ukHD3P3hpavGHdAbUUDDRvhWj8IRaS/cn0LfUF2dqQQNi2negdk+9M02/Qkhs6ezPSDswUYUZaanvCOnHZtk6tMN3mjcTB0mvCld+SwLR5cP+7sc6DtiOIm4yXQfSlu2fB6Lv+scv1+IWGEG4QKqhdMe+V2RsIdTO9Ak5Swr2wrRthnarvEjQamQV8mdr60AbSul5FBNzUGQz1vYIxWbZ+B//duDkT5jz3XG7rMg4g/ksOKc0dvhJ4A4CTZLH/PyFNWZATugaLgmIAJMAfTUd5RgQytB9OZpnO9sQLM+VgtcltgCsqj27ZQVEsFBy+r1bpwUn5ceM9NTT5B4pfqwH3Q0jsmYBjmYGLSWwcmmQzzrET6oeWyCI9GBlB/H4ZLDNtkn4IGih/IeRZpH9vhzb/IGyvAQoK/TeA9aHM2ltpwiV/mjaMZJrJMpFmJxCqYujCipDo64oArJswp4EAqW1rbwaW7eB1fl+WDTN6vMK3M/44Z+EjMadDazkPXhrTwCZLnoW0D5oMNDQ0No4j/AIXcE7TAr/YkAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAABIUlEQVR4Xt2TLUhDURSAj6IzTE3iT7KJwSSCw2oSi0URuzKDWAQRm2gRi7CgxT7GugaxGEyCFhEsNpNFQcSwfYdzGdfzYL5r3AdfeN+9nL13955IxzOARR/zcozf2MAdt5bEhtiQkl9I4Rw/sMcvpPCMlz6mMCb2KLs4jAs4/mtHDlbFhlSxhpv4gkfxpr84ExuyHzUd8I7dUWvLE966diE2xDOJiz6OSPYulFesu6aUcd3HFbEhc1GbDW0tam2p4Cf2Ru0Uv7Afl3EpdL2LA+wK1y0e8cq1O7wW23wj9j3pOUyLvU96Li0K+CPZZ9zCB7FB86HN4JTYWWX+sVEfAkM46NoJHrqWhP76G07gnlvLjZ7PPW6Lnc2/0S+8z8cOogkRly+6oTL4ZAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAiCAYAAADiWIUQAAAD70lEQVR4Xu3cTYjVVRjH8VNqGystkTLcVGKZQRstUcOFSBiaurB2GRjiRjLxFYR8l8CFZqI70SLNNxQjisiNkkKQL6QgqRQhqYhFYYIp9fz4nzP3uQ9zZ+YydwZivh/4cc95/v/5z713dTgvNyUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPq0p2KhhYbGAgAAgNy3jInFHjTe8nYs9pI5sdCkZywjY7EJ3zfId5aXLNMtV9ruBgAAyC7GQg96wPKT5W68YPrFglkZC03aH/pvhX6zrof+ess/odZIf8vjrv9brhXl8z9nGe3qAAAA6d9Y6EGvWg6m9v/nlFgwJ2KhSXHA1l0fh/5Jy6lQa8R/vudT/XfwoGvLltAHAAB93L1UDUQWW6aGa6122fKw5ZZliKt/Zblm+TrVZsE0WNOgRrXiE8sMy07LKMtWy++WdyxLLcctAyyvp+rvyjN17xepfpB0wbLWssPyQ66dSdU9my0fWG7nugy3THN9PVf3ajmz2e/t89T+oLXo6BoAAOhjnkzVsl5xyLXfde1WmZ1fX7N84y+YfaH/QqofuGggpgHZwJw7uT7TMj+3tWlfg7AizrCV571vGevqGuCVfXUatJXlyW2urf1vD+V2oRk2TzNnXaH38WUsOgzYAABAmzctE3L7acsCd+2oa7eClv20X+6c5XyqZvb8qcjOBmwaDGmGbnfOrlx/I9VmvjRrty635YBrS3nep5YRrj4x1ZYhy2ybaAav7DNb5OqiQePGUCsDx87ofSyJRYcBGwAAaPOja5/Nr8NSNTga5655OgjQKCvcfdH20Neg5Jjr782v3+ZXv89L+790YMFv+tesmGiJtCxJalO/nzE8bHnF1crzNHtY/l5WW17O7dOu/lGqZvZEJzj9d7LB8khu65CABmtrapfrDhh4+hx/pvYPWRQ6uQsAAJAmW/52fe2r0mDpMcuzlrnuWnfppys0SFmY+1pe/CvXys9YDMptP/OlQdAR19d71jLkZ5bBqZpB0zOUZZarqXpuoefpc2nQ9Uu+T6+i96IDAzqtuTzX1NY9er1k+cNyM1+TVa49KVV76TRbJy+m2rNlT6r/DvX7bT+n6vlazv01VbOa7elouRQAAPQhmuGJP9Ra9mhpafFRf6GXaE9d5A8niH4HTYO7VtBzyixZV+jggveEa29K9cuxMi/0u0JLx7NiEQAAINJJzTIbhhoNyuJPcBTa+/ae6/tDDV2lWcMbsQgAAIDmxIMGrfRhajwgBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8f/0Huoei9MkhTTAAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAZCAYAAAABmx/yAAAA+0lEQVR4XmNgGAVEAxcgXoQuSAw4CsTH0QUJAR4g/g3EHegSuAAfEKsAcQwQ/wfidCBWBWJWZEXYQBYQ7wTiZwwQG0FsEJZGVoQPgPx3DF2QEOAG4l9A3I4ugQQEgTgJXZBsjW4MkIBxR5dAAnZAPBFdsI0BEjCgKCEJgALlJBJ/HhCzQNmMQJwGxP1ALApXAQUvgHgOlA1SVIIkB/KXMhBPQRMHgwogfg/EM4G4FU3OBEqfAGIvZAkYADkDlx9B4j8YIKmMJAAK7dPogsQAUPx2oQviA7CQBYU4vjhGAQ5AfBWIvYH4EBAzo8jiASJA3AjEhUDMiSZHPgAA9MsmsrAIZLEAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAXCAYAAADQpsWBAAAAyklEQVR4XmNgGAVYgQsQL0IXJASOAvFxdEF8gAeIfwNxB7oENsAHxCpAHAPE/4E4HYhVgZgVWRE6yALinUD8jAFiE4gNwtLIinABkH+OoQviA9xA/AuI29EloMAdiHXQBdsYIE4DBQbRAOSsk0j8eUDMAsRcQNwMxLFIcnDwAojnQNlpQFwCZRcDsTJUHgNUAPF7IJ4JxK1I4iZAnAPEi5HEUIAoA3Y/nWGAJC+igRYQPwZiQQZIfBIF5BkgAVTNgN0VOAEoDocSAAD36R66/ryQQAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA1klEQVR4XmNgGLSAF4i50QXRQTsQfwfi/0BciiaHFaQwQBRboktgA7OB+CsQs6JLYAN3gHg3uiA2IMsAcUINkpg4EFsh8eEglgGi2AaImYC4E4iXAPFFIA5FUgcGc4D4GwMk2KYAsRkQZzFADIhDUgcGIPdeZYBo0oKKqQNxNhBzwBSBQDQDxARrBogb3wDxXGQFyGAmA2qQbQDiW1B2JBC7Q9lgcA2IdyHxVwHxASh7CwNS9LMD8V8gzoEJAIELEL9mgGjC8JwcEDOiiYE8JYomNgrgAACXXCOZ5tyyogAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAYCAYAAAB0kZQKAAABbUlEQVR4Xu2VvSuFYRTADyHyNUkoExuTjdQtYmAzGChFUTIalFIGH4vF5iPJcP8EESuS0YLdx4bBwsDv9DzX+9wj1+XmrVv3V786zznvcO57zvNekQIF8oBe3LPJuDnBM5uMkyp8w1VbiIMabMFRfMcpbMXS8KH/ZhoP8U7cm9BYbQofigvdh1ObjJNKfMUVW4AxvMIZW/gl9ThpkyF94vah3xY819hpk4YGbLZJzzoe4K0thCyL2we9IZZGfMEyWzB0ifsx35GQH5rQXTgPzjtY4uMRPPbxEC5ihT+HdEuOTTzgto91brNBbRMXfF5H8iTuCltybmIOH3EDl0ztBi+wHYuwzed10eYDt3DX5MJrnhD3GchInXzdidQ+DOAldqSXpTywBwdNTptOkcD74Jw1w3jkY/1j06+pXtXazycishnHn5pYk2g/JjCJ41E5jUxN6DLv47O4kekbyxodT3Fwrg5iiy6tjqRAfvMBefs/EezQtyoAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAAB/UlEQVR4Xu2VPUgdQRDHR4wWmvgBioggD1EMCFoIKTRgI1EbO4uEFGmMRMRK0MLimmhiJ4KgxI/4jZ1YJb0mFkkVsEiIQkBtNJ2FCvqfzJ6Mo++9E989ItwPfnA3s+euu7PziCIiIu4FzXDOBtNADK7BIhMPzAb8YoNp4C08hbk2EYSHJB+/s4k0wJv11QaTkQcr4Ut4DrtgFczSg0KA/z7PWwNP4BTJvPl6UCK64Se4R7Lj/MyW6UEh8JRknh8kG7bp3tv0oCBwffPH6eZO9c0f8XEN24TjMeywwRTBm3VTQyiE6/A3zDC5S56RHFeLiXO5fITf4KLJpYJkG/YcfrZBzRDJcXFnuQmPwlm4v2GtNuGYhIM2qOHj2lLv0/CBevco/sKLSbqDJhvWmxjDMd2teMPO4CP3PkDSZXx+klziUtgP21XuHwfwg3t+DftUjvEo/sJ3SE4rpmKzJDvJLdbnlYv58zAr8Jd7roOrKsdlegxrYSfJJb62Bv5P/8IJkgEWDy7ZoIPj3NJ0mfXAP/CJijW42BsVayJZ3AxcIPlN8XkBD+EIyeUspzg9no88UY0v22CKyIElNkhS36NwDM6bXGA8Cm/h8diGjbAC7pNsau+VEQngXjoOv8NdklKq1gNCgkvmiOQic5Pg+d/DAj3of0XXeyZJSUVE3JYLiT9aBC9+Oh0AAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAYCAYAAABjswTDAAABvUlEQVR4Xu2WTShEURTHj49iw0KJRI1IvooNsnsbKxaysRGKpKRQsrCyVEqUhY/FlIWUNSJZkVDWilK+wgoL2eB/OvfldjRjxpvemHq/+jX3/c+b5nTn3vseUUBAatIMT2LwEO7DOvlacliGZ7AVlsBCuAM/YTvMN3mPycrla/6TCc9hnsrv4StJ3eYGZqjMN1rgtMoqSWZwS+Xp8FRlvtILy1Q2SNLshMpz4KTKks46SbONuvAfeYAvlMS1GSs1JLO6qQsWBXBAh3FSCldITqExK+cTaQ8ekCzRqAyRNDuuC4Z5uA1vdeEP1MMj+AizrHwKNljXEdkgaTbazQ4lptkRWAufYbeVL1jjiKTBJ/p9vTqUmGZnzeccPDZj/l03jwrvfp7VXV1QOOS9WT6zZ8y4An7AJpIeutybNEXwEl6RzOib8RpekGwEjQPvdBgnvF758e3C+2CV5GwvtnLPOCSPYy+MwpB1zafAO1yzsoTg0M9mQ3DYuu6H1WacTTJjud9lWrTGDC8L/ieXVO4JPlb4DOYdHIZtJu8geSFyNya/UvaZMb+x8RrnnV9F8qrJsxg2dRee7U6VBQSkLF/mHVcCHdhUngAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABF0lEQVR4Xu2TvUoDQRRGr1qZSkTSpBQMIjZ5AAvjA0jE0jYJ1jZ2go1VQLCxsgkk8SUELRQUkQTSBEJIo502kkLUnHFm2J27m8J+Dxx25n7D/C0jkvFftnGIr/iGzTD+4xFHOBA7thGkijZ+4A+uqmwBT/AWC2GUpItH+CvpK57hvi5qiniNS/iJ75gLRojcYV7VEtTw0LUvxe6qGsWyiM+x/kxauO7am2InMkf1lPEi1p/Ji+rfiJ1sy/VPcS+K0/H3E6cidiJfN/ezEsXp1CW6H4/53WP8wjV8CuN0Orihi3Asdlf3eK6yBHPYd1+NOcpE7GS7KktwgD2c14HjCr9xWQeeHbHvyqxoNE/DvDlNCR90MSMDpvMbNCf6RtASAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAYCAYAAAB9ejRwAAABrElEQVR4Xu2VSyhEYRTHj2feK3mkWCtLRSRTioiyIbETysJOKUosbOzIwqNkRZKdxMIjSbZsRDYW2Cl5JQv+p/Pd5syZvGbkFvOrX833P1/Tued+916iGH+MTJhuQ78Yh8/wFfabmq/0kDRVbgt+MgPvYKIt+Mkp3LChn+ST3LoBmAPrYVHIDh9oI2lqGa7AXngOx/Sm32aapKkhlXFDNzBeZb/KCdw32TxJU5oyeAAXTf5dckme9nfhDXZKzAVcNRnDT+mgDQ18Rgtt6JgkeaAubUHTStJUhcp4Ipy1q8zjDFba0MD1WhsqAvRJU1PwHiapbAI+wQzYAptdXgAfYTIshSOw2NU0VRRlU8dw02SHcAvGwR0Kfg95cpzXwQa4C7tdTRNVU3zFLxT+x33wiKSBGpXPkrwqOtyap8RfAD6XfCY95+CCyXjKHgF4pdZh5NnAkQ2zTMbnqQtuU/iFpCj5QhpNxlP3CMBrtY4Ybv6B5Ox1wiWSA12tNzm+cvt+pKkmuOZ+l8A9OBwsh/BRU6NwHd6S3GKeaMTwhFLVOg0mqLWGXy/6LMb4H7wBEJlNtebGpisAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAA00lEQVR4Xu2QPw5BQRCHp6AiEoVKSziIcAKdmkT9CuU2/pxAIpGXkOAMDoAoOMOrRKen4DfZRyYT1oaSL/mS3d/MZN88oj9fUYFTHfqyghsd+pCGFzjQBRcZWIANeIUtWIRJ2fSKNlzCA9mX+czmZdM7eN+1Dn1IwTPs60JMGdZ1eKdKdt+ayvnTJ3AHZ6r2oEd2X/7jzzDkGOZdt+IewoS4G3IMH+E4PjdhIGqMIcdwB57gCHZVjTFwrkNJjtw7L3Toi6EPhrNwCPcwIrtWSTb8GjccriS5oYq0bgAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAADJUlEQVR4Xu2XWahNYRTHl3kmkVkypSgPZMiUUoZERDxcdK8hXswlIu6bFInwgAdj8kCilOFBMsYDkgdzQqYilHn4/8/au7P22vvcs89x7i11f/XrnrPWd/b9zv7Wt/Z3RGqpFjrCVbAMtnS5Gqc1bOKDKegCd8FecDN8DBuZPK/b1LwvCF5oFJwoerfy0Qleh618IgXr4K3gdWP4GS7OpqUvvASbmVhe+sCT8B6shBvhR3gGts8Oi8AvfQ1W+ERKesDx5v07iV9ri+i86rp4ImvgKzgH1jfxbvAGfAKbm3gIy4H5Oj5RBJPhU9FysnDfvIYLXDzGVvgNDvaJgHL4B25wcf4DrtgwFy8GrjhLqLtPBMyFz2BDnwiZIjrJnT5h4B7gmJsuvhTedbFiYP0fFl19ruzQaDpDPfgdTvUJws35An4RnWwuWsDf8L2Ln4MnXCxkJOxv3veEEyTeEFjG++FYOADOhAsjI7Lchzt8kKwQvdO7fcLB0uG4Ry7+UHSPeLjfOLlfohuX7XW76LOCXWlMdmimXHlt62iTt5yGt32QnBf94HyfcPAOcdwpEwuXepGJEfb848Hrl/ArHJ5NZ8rzqHlfCLwZb32wgegG5wSHuJyHX4DjlpkY65kxe3cJG8Y02FmSGwRL+YiLpWWJ6CpH2jD7P+ueJrXVkH6iY56LPrBCWM+cKP8mweMG8wNNjA83xnLtgXzMFp1LrHOx5nnhqjY6a51jeBEL2yTj0108ZA/8IFqCIVydn7CdiRXCetEVjXFAdDIzfCKAG5X/mEvqYSfjZ1f7RMADie4pwhPD2eD1NtjW5NLA+fJZE4PnKLbUq7CNy82DP6TqpynLba8Pil6XX3K5ibHtMrZStBUfM7m0XBH9MokMgndEWynv/CbRfs07N8KMS2IfvOBiZJxom+3t4jwvXYYXRZtBobwRbeE54UOJm3KW6ImXHSkN3B9sh7YJEF4vV9nw2M6OWShdRbtsB58oBWyDrPu1PlEN8AhzyAdLyST4SYorlbTwgcq97Eu15PDH0UEfLBFcdf5M4DmtRqiU3PviX+BJmEf4Wv4L/gItXZiAxZdPyQAAAABJRU5ErkJggg==>
