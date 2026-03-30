import { useState } from 'react'
import Header from './components/Header'
import Tabs from './components/Tabs'
import Overview from './components/Overview'
import Blog from './components/Blog'

export default function App() {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="container">
      <Header />
      <Tabs activeTab={activeTab} onTabChange={setActiveTab} />
      {activeTab === 'overview' && <Overview />}
      {activeTab === 'blog' && <Blog />}
      <div className="footer">bisratz/architectLLM</div>
    </div>
  )
}
