import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import CitizenUpload from "./pages/CitizenUpload";
import AdminDashboard from "./pages/AdminDashboard";

export default function App() {
  return (
    <BrowserRouter>
      <nav className="top-nav">
        <span className="brand">🛣️ PRISM</span>
        <Link to="/">Report a pothole</Link>
        <Link to="/admin">Admin dashboard</Link>
      </nav>
      <Routes>
        <Route path="/" element={<CitizenUpload />} />
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
