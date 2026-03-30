export default function FineTuningPost() {
  return (
    <div className="blog-post">
      <div className="date">March 2026</div>
      <h2>Learnings from fine-tuning GPT-OSS 20B</h2>

      <p>
        I fine-tuned OpenAI's GPT-OSS 20B to be better at system design reasoning. It took
        three attempts. Two of them failed. Here's what I learned.
      </p>

      <h3>Attempt 1: The Unsloth template</h3>
      <p>
        My first approach was the path of least resistance: Unsloth's pre-built notebook for
        GPT-OSS fine-tuning. Unsloth is a popular framework that promises 2x faster training
        and lower memory usage. The notebook ran on a free T4 GPU (16GB VRAM), which meant
        everything had to be aggressively compressed to fit.
      </p>
      <p>
        The model loaded in 4-bit quantization, which crushes the precision of every weight in
        the network. Think of it like saving a high-resolution photo as a tiny, heavily compressed
        JPEG. The LoRA rank was set to just 8, meaning the "adapter" that learns the new skill
        had almost no capacity. And the sequence length was capped at 1,024 tokens, which meant
        the model never saw a complete system design answer during training. Every example was
        truncated mid-thought.
      </p>
      <p>
        The worst part came at the end. The notebook merged the LoRA weights back into the
        quantized base model and then saved the result as 16-bit. Going from 4-bit to 16-bit
        doesn't restore the lost precision. It just makes a bigger file that's still blurry. The
        merge step introduced numerical errors on top of already-degraded weights.
      </p>
      <p>
        The result: a model that gave generic, surface-level answers. It would list components
        like "use Redis for caching" without reasoning about why, what the trade-offs are, or
        how to size things. Worse than useless for system design, where the reasoning is the
        whole point.
      </p>

      <h3>Attempt 2: Pure HuggingFace (with bugs)</h3>
      <p>
        Someone recommended I try a different notebook that used the standard HuggingFace stack
        (transformers, PEFT, TRL) instead of Unsloth. The approach was smarter in theory: higher
        LoRA rank (r=64), longer context, bfloat16 precision.
      </p>
      <p>
        But the code was riddled with bugs. The tokenizer's <code>apply_chat_template</code> was
        called incorrectly, returning a dictionary when the code expected a tensor. The training
        crashed with an Arrow serialization error because the trainer couldn't handle the nested
        JSON structure of the chat messages. The save step used a parameter
        (<code>safe_serialization</code>) that PEFT models don't support. And the merge step
        tried to create a full copy of the 20B model in GPU memory that was already full.
      </p>
      <p>
        The weights that did get pushed to HuggingFace were corrupted. The merge-from-quantized
        pipeline had the same fundamental flaw as Attempt 1, plus all the code bugs on top.
      </p>

      <h3>Attempt 3: From scratch</h3>
      <p>
        For the third attempt, I threw away both notebooks and built the training pipeline from
        scratch using the official OpenAI fine-tuning cookbook as reference. The key decisions:
      </p>
      <ul>
        <li>Load the model in native bfloat16. No quantization at all. Start from a
            perfect copy of the original weights.</li>
        <li>Use LoRA with r=64 (8x the capacity of Attempt 1) so the adapter has
            enough room to learn the patterns of system design reasoning.</li>
        <li>Set sequence length to 4,096 tokens so the model sees full answers,
            not truncated ones.</li>
        <li>Pre-format the dataset to plain text before passing it to the trainer,
            completely avoiding the Arrow JSON serialization issue.</li>
        <li>Save only the LoRA adapter (about 100MB), never merge it back into the
            base model. At inference time, load the original base model and layer the
            adapter on top. No lossy merge step, no corruption risk.</li>
        <li>Run on an H100 80GB so there's no need to compromise on quantization
            or batch size just to fit in memory.</li>
      </ul>

      <h3>The core lesson</h3>
      <p>
        The single most important thing I learned: quantization before training degrades
        the starting point, and merging after training degrades the endpoint. Both previous
        attempts did one or both of these things.
      </p>
      <p>
        The fix is conceptually simple. Start clean (full precision), train only the adapter
        (LoRA), save only the adapter (no merge). The original model is never touched, never
        compressed, never corrupted. The adapter is a small, clean diff that gets layered on
        at inference time.
      </p>
      <p>
        Fine-tuning isn't hard. But the gap between "code that runs" and "code that produces
        good weights" is enormous. Every parameter choice, every data preprocessing step, every
        save/load decision either preserves or destroys the information your model needs to be
        useful.
      </p>

      <h3>Why not Unsloth?</h3>
      <p>
        Unsloth is a good tool for what it's designed for: fast, memory-efficient fine-tuning on
        consumer GPUs. But for a 20B parameter model where output quality matters more than
        training speed, the compromises it forces (4-bit quantization, aggressive memory tricks,
        proprietary merge pipeline) work against you. The weights can only be loaded back through
        Unsloth's own tooling, and the merge-to-16bit path is lossy.
      </p>
      <p>
        Using the standard HuggingFace stack means every component is well-documented, every
        parameter is inspectable, and the outputs are compatible with any inference framework.
        When something goes wrong, you can debug it. When something works, you understand why.
      </p>
    </div>
  )
}
