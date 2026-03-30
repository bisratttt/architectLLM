import { useState } from 'react'
import FineTuningPost from '../posts/fine-tuning'
import TrainingDataPost from '../posts/training-data'

const posts = [
  { id: 'training-data', date: 'March 2026', title: 'How the training data gets made', component: TrainingDataPost },
  { id: 'fine-tuning', date: 'March 2026', title: 'Learnings from fine-tuning GPT-OSS 20B', component: FineTuningPost },
]

export default function Blog() {
  const [activePost, setActivePost] = useState(null)

  if (activePost) {
    const post = posts.find(p => p.id === activePost)
    const PostComponent = post.component
    return (
      <div>
        <button className="back-link" onClick={() => setActivePost(null)}>
          &larr; all posts
        </button>
        <PostComponent />
      </div>
    )
  }

  return (
    <div>
      {posts.map(post => (
        <div key={post.id} className="blog-index-item" onClick={() => setActivePost(post.id)}>
          <div className="date">{post.date}</div>
          <div className="title">{post.title}</div>
        </div>
      ))}
    </div>
  )
}
