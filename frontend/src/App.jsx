import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import CitizenUpload from "./pages/CitizenUpload";
import AdminDashboard from "./pages/AdminDashboard";
import AdminLogin from "./pages/AdminLogin";
import ProtectedRoute from "./components/ProtectedRoute";

function TopNav() {
  const location = useLocation();
  if (location.pathname === "/admin/login") return null;

  return (
    <nav className="top-nav">
      <span className="brand">
        <span className="brand-mark">🛣️</span> PRISM
      </span>
      <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
        Report a pothole
      </NavLink>
      <NavLink to="/admin" className={({ isActive }) => (isActive ? "active" : "")}>
        Admin dashboard
      </NavLink>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <TopNav />
      <Routes>
        <Route path="/" element={<CitizenUpload />} />
        <Route path="/admin/login" element={<AdminLogin />} />
        <Route
          path="/admin"
          element={
            <ProtectedRoute>
              <AdminDashboard />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
