import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import CitizenUpload from "./pages/CitizenUpload";
import AdminDashboard from "./pages/AdminDashboard";

export default function App() {
  return (
    <BrowserRouter>
      <nav className="top-nav">
        <span className="brand">🛣️ PRISM</span>
        <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
          Report a pothole
        </NavLink>
        <NavLink to="/admin" className={({ isActive }) => (isActive ? "active" : "")}>
          Admin dashboard
        </NavLink>
      </nav>
      <Routes>
        <Route path="/" element={<CitizenUpload />} />
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
