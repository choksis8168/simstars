import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { NewSession } from './pages/NewSession'
import { RunDetail } from './pages/RunDetail'
import { SessionDetail } from './pages/SessionDetail'
import { SessionList } from './pages/SessionList'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SessionList />} />
        <Route path="/new" element={<NewSession />} />
        <Route path="/sessions/:sessionId" element={<SessionDetail />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
      </Routes>
    </BrowserRouter>
  )
}
