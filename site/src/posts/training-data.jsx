export default function TrainingDataPost() {
  return (
    <div className="blog-post">
      <div className="date">March 2026</div>
      <h2>How the training data gets made</h2>

      <p>
        The model is only as good as the data you train it on. For architectLLM, I needed
        thousands of system design conversations that teach the model to reason from first
        principles: identify the right primitives, analyze constraints, do back-of-envelope
        math, and explain trade-offs. That kind of data doesn't exist in any public dataset.
        So I built a pipeline to generate it.
      </p>
      <p>
        The pipeline has six phases. It starts with real engineering blog posts and ends with
        balanced, quality-filtered training examples in the format GPT-OSS expects. Here's
        how each phase works and what I learned building it.
      </p>

      <h3>Phase 1: Extracting primitives from blog posts</h3>
      <p>
        The raw material is engineering blog posts from RSS feeds. Posts from infrastructure
        teams at various companies describing how they solved real scaling problems. The
        extraction phase fetches these posts, converts the HTML to markdown, and filters out
        anything under 300 words.
      </p>
      <p>
        But raw blog posts aren't training data. A post about how someone built a rate limiter
        contains useful architectural knowledge, but it's buried in narrative, company-specific
        context, and implementation details. Phase 1 uses GPT-4.1-mini as a "teacher model" to
        read each post and extract a structured annotation: the primary system design primitive
        (from a taxonomy of 43 patterns like sharding, circuit breakers, pub/sub), the
        constraints that drove the design (latency SLOs, throughput requirements), the
        trade-offs considered, failure modes, and a domain-independent lesson.
      </p>
      <p>
        The temperature is set low (0.2) because I want consistent, accurate extractions, not
        creative ones. Posts that aren't about infrastructure (hiring posts, culture pieces,
        frontend UI work) get flagged with a SKIP marker and dropped. The output is one
        structured annotation per blog post, grounded in what the post actually describes.
      </p>

      <h3>Phase 2: Generating questions with Evol-Instruct</h3>
      <p>
        This is where one blog post becomes many training examples. For each annotation from
        Phase 1, the pipeline generates 8 questions that apply the same primitive to different
        domains. If the blog post was about rate limiting in an API gateway, the questions might
        ask about rate limiting in a gaming backend, a healthcare data pipeline, or an IoT
        sensor network. The idea is to teach the model that the same architectural pattern
        applies across domains with different constraints.
      </p>
      <p>
        The questions follow a specific complexity distribution: one focused on a single
        primitive, four or five that require combining multiple primitives into a full system,
        one that asks about adapting the pattern across domains, one about trade-offs against
        alternatives, and one about failure modes. This distribution is deliberate. Most system
        design problems in the real world aren't about one pattern in isolation; they're about
        combining several patterns under constraints.
      </p>
      <p>
        The question generation uses a technique called Evol-Instruct, which evolves questions
        through three strategies: deepening (adding constraints like latency SLOs or compliance
        requirements), widening (combining with secondary primitives), and concretizing
        (embedding specific numbers like 10M daily active users or 99.99% uptime). The
        temperature here is high (0.8) because diversity matters more than consistency. I'd
        rather have varied, creative questions than eight versions of the same prompt.
      </p>
      <p>
        The pipeline also generates multi-turn conversations: a question followed by
        increasingly specific follow-ups that push the model to go deeper on particular
        aspects of the design.
      </p>

      <h3>Phase 3: Two-pass response generation</h3>
      <p>
        Generating the responses is the most expensive phase and the one where quality matters
        most. Each question gets a two-pass treatment. The first pass generates a chain-of-thought
        analysis: the model reads the original blog post excerpt alongside the question and
        produces structured reasoning grounded in the source material. What constraints does
        the question impose? What primitives are relevant? What trade-offs should be considered?
      </p>
      <p>
        The second pass takes that analysis and generates the final response: a structured
        system design answer that follows a specific format. Restate the challenge in terms of
        primitives. Analyze requirements (functional, non-functional, scale). Lay out the
        high-level architecture with component responsibilities. Deep-dive on 2-3 critical
        components. Discuss trade-offs and alternatives. Cover failure modes and mitigation.
        Describe the scaling path (what changes at 10x and 100x).
      </p>
      <p>
        The two-pass approach exists because single-pass generation tends to produce answers
        that sound confident but aren't grounded. By forcing the model to first analyze the
        problem against the source material (at temperature 0.3 for accuracy) and then write
        the response using that analysis (at temperature 0.5 for fluency), the final answers
        are both technically grounded and well-structured.
      </p>
      <p>
        One important rule enforced through the system prompt: describe patterns, not products.
        The model should say "an in-memory key-value store with TTL-based eviction" not
        "use Redis." This keeps the training data vendor-neutral and forces the model to
        reason about why a pattern works rather than just name-dropping a technology.
      </p>

      <h3>Phase 4: Quality filtering</h3>
      <p>
        Not everything that comes out of Phase 3 is good enough to train on. Phase 4 is a
        multi-stage quality gate. First, semantic deduplication: the pipeline embeds all
        questions using Qwen3-Embedding-0.6B, computes pairwise cosine similarity, and flags
        anything above 0.94 similarity as a near-duplicate. This catches questions that are
        technically different strings but semantically identical.
      </p>
      <p>
        Next, exact deduplication via SHA-256 hashing of normalized question text. Then a brand
        check: a regex-based scanner that flags responses mentioning specific product names
        from a blocklist. Responses that say "use Kafka" instead of "use a distributed
        commit log" get filtered out, unless the brand appears in an acceptable context
        like comparing alternatives.
      </p>
      <p>
        Length filtering removes any response under 50 tokens (too short to contain meaningful
        reasoning). Finally, an optional LLM judge pass using GPT-4.1-nano scores each
        response on five dimensions: technical accuracy, completeness, structure,
        actionability, and primitive coverage. The threshold is 3.5 out of 5. The judge
        runs at temperature 0.1 for maximum consistency.
      </p>
      <p>
        The filtering is deliberately aggressive. It's cheaper to generate extra examples and
        filter hard than to train on mediocre data. The pipeline logs a detailed breakdown of
        why each rejected example was rejected, which is useful for tuning the generation
        prompts upstream.
      </p>

      <h3>Phase 5: Coverage validation and composition</h3>
      <p>
        Passing quality checks isn't enough. The dataset also needs to be balanced. Phase 5
        builds a primitive-by-domain coverage matrix and validates three rules: every primitive
        must appear in at least 4 domains (no single-domain overfitting), the top 10 most
        important primitives need at least 50 examples each (sufficient signal for the patterns
        that matter most), and no single domain can exceed 25% of the total dataset (prevent
        domain bias).
      </p>
      <p>
        The composition phase then samples from the filtered pool to hit a target distribution:
        30% single-primitive exercises, 25% full-system designs, 20% cross-domain questions,
        10% trade-off comparisons, 10% failure-mode debugging, and 5% multi-turn conversations.
        Within each category, sampling uses round-robin across primitives and domains to maximize
        diversity, with quality score as the tiebreaker.
      </p>

      <h3>The final format</h3>
      <p>
        The export phase converts everything into the Harmony format that GPT-OSS expects for
        fine-tuning. Each training example is a conversation with four messages: a system prompt
        setting the architect persona, the user's question, an analysis channel containing the
        chain-of-thought reasoning, and a final channel containing the structured response.
        The chain-of-thought is preserved as a separate channel so the model learns to reason
        before answering, not just produce answers.
      </p>
      <p>
        The final output is about 5,000 examples in a single JSONL file, plus metadata tracking
        the distribution of primitives, domains, and complexity levels. The whole pipeline is
        resumable at every phase (it tracks processed IDs), so if an API call fails or a quota
        runs out, you pick up where you left off instead of starting over.
      </p>

      <h3>What I'd do differently</h3>
      <p>
        The biggest lesson: invest in the filtering pipeline early. My first version had minimal
        filtering and the training data was noticeably worse. Semantic deduplication alone
        removed about 15% of examples that were near-copies of each other, and the brand check
        caught responses that would have taught the model to recommend specific products instead
        of reasoning about patterns.
      </p>
      <p>
        The two-pass response generation was the single biggest quality improvement. When I
        switched from single-pass to analysis-then-response, the average LLM judge score
        jumped noticeably. The model produces better answers when it's forced to think
        before it writes, which isn't surprising, but it was good to see it confirmed
        empirically.
      </p>
      <p>
        The Evol-Instruct technique for question diversity was worth the complexity. Without
        it, the questions cluster around obvious formulations. With it, you get questions
        that approach the same primitive from genuinely different angles, which gives the
        model a richer understanding of when and how to apply each pattern.
      </p>
    </div>
  )
}
