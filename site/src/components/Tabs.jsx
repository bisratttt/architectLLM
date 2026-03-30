const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'blog', label: 'Blog' },
]

export default function Tabs({ activeTab, onTabChange }) {
  return (
    <div className="tabs">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`tab ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
