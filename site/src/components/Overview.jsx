export default function Overview() {
  return (
    <div>
      <a href="https://huggingface.co/bisratz/architectLLM-lora" target="_blank" rel="noreferrer" className="hf-link">
        View on HuggingFace &rarr;
      </a>

      <div className="section">
        <h2>What is this?</h2>
        <p>
          architectLLM is a LoRA fine-tune of OpenAI's GPT-OSS 20B, trained specifically on
          system design and software architecture reasoning. The training data covers the kinds of
          problems you encounter when building and scaling real production systems: designing rate
          limiters, building distributed caches, choosing between consistency and availability, sizing
          infrastructure, and reasoning through trade-offs that don't have clean textbook answers.
        </p>
      </div>

      <div className="section">
        <h2>Training details</h2>
        <table className="info-table">
          <tbody>
            <tr><td>Base model</td><td>openai/gpt-oss-20b</td></tr>
            <tr><td>Method</td><td>LoRA (r=64, alpha=64)</td></tr>
            <tr><td>Precision</td><td>bfloat16 (native, no quantization)</td></tr>
            <tr><td>Learning rate</td><td>2e-4, cosine schedule</td></tr>
            <tr><td>Sequence length</td><td>4096 tokens</td></tr>
            <tr><td>Framework</td><td>HuggingFace transformers + PEFT + TRL</td></tr>
            <tr><td>Hardware</td><td>H100 80GB (Google Colab Pro)</td></tr>
            <tr><td>Training data</td><td>1,787 system design conversations with reasoning</td></tr>
          </tbody>
        </table>
      </div>

      <div className="section">
        <h2>What it's trained on</h2>
        <p>
          The training set focuses on common system design interview problems and real-world scaling
          challenges. Each example includes structured reasoning: the model learns to identify the
          right infrastructure primitives, analyze constraints like latency, throughput, and
          consistency, perform back-of-envelope calculations, and explain trade-offs explicitly rather
          than just listing technologies.
        </p>
        <p style={{ marginTop: 12 }}>Topics include:</p>
        <ul style={{ marginTop: 8 }}>
          <li>Rate limiting and API gateway design</li>
          <li>Distributed caching strategies</li>
          <li>Message queues and async processing</li>
          <li>Database sharding and replication</li>
          <li>Load balancing and service discovery</li>
          <li>Consistency models and conflict resolution</li>
          <li>Back-of-envelope infrastructure sizing</li>
        </ul>
      </div>

      <div className="comparison">
        <div className="comparison-header">Example: Rate Limiting</div>
        <div className="comparison-prompt">
          "Design a rate limiting system for an API gateway handling 500,000 requests
          per second across 200 microservices."
        </div>
        <div className="comparison-grid">
          <div className="comparison-col">
            <div className="label">Base model</div>
            A rate limiter controls how many requests a user can make. You can use a
            token bucket or sliding window algorithm. Store counters in Redis. Set limits
            per user. Return 429 when exceeded. Use middleware to check limits before
            processing requests.
            <br /><br />
            <em style={{ color: '#555' }}>Shallow. Lists components but doesn't reason about
            the constraints: 500K RPS, 200 services, distributed coordination,
            clock skew, or failure modes.</em>
          </div>
          <div className="comparison-col after">
            <div className="label">architectLLM</div>
            At 500K RPS across 200 services, the core constraint is coordination latency.
            A centralized Redis counter adds ~1ms per request, meaning 500K RPS needs ~500
            concurrent connections just for rate checks. Two-tier approach: local token
            buckets per service instance (refilled via async sync with a global ledger)
            handle the hot path with zero network calls. Global Redis counters are the
            source of truth, synced every 500ms. Trade-off: you accept up to 0.5s of
            over-admission during sync gaps. For 200 services at ~10 instances each,
            that's 2,000 local buckets syncing with ~50 Redis shards. Back-of-envelope:
            2,000 syncs * 2/sec = 4,000 Redis ops/sec for coordination, well within
            capacity.
            <br /><br />
            <em style={{ color: '#4a8' }}>Reasons through constraints, calculates sizing,
            names the trade-off explicitly.</em>
          </div>
        </div>
      </div>
    </div>
  )
}
