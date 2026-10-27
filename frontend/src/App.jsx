import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import CitizenUpload from "./pages/CitizenUpload";
import AdminDashboard from "./pages/AdminDashboard";
import AdminLogin from "./pages/AdminLogin";
import ProtectedRoute from "./components/ProtectedRoute";
import Logo from "./components/Logo";
import { CameraIcon, ListOrderedIcon } from "./components/Icons";

function TopNav() {
  const location = useLocation();
  if (location.pathname === "/admin/login") return null;

  return (
    <header className="top-nav">
      <div className="top-nav-inner">
        <Link to="/" className="top-nav-brand" aria-label="PRISM home">
          <Logo tagline />
        </Link>
        <nav className="top-nav-links" aria-label="Main">
          <NavLink to="/" end aria-label="Report a pothole" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            <CameraIcon size={16} />
            <span>Report a pothole</span>
          </NavLink>
          <NavLink to="/admin" aria-label="Admin console" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            <ListOrderedIcon size={16} />
            <span>Admin console</span>
          </NavLink>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  const location = useLocation();
  if (location.pathname === "/admin/login") return null;
  return (
    <footer className="site-footer">
      <span>PRISM · NMIMS Hyderabad capstone</span>
      <span className="site-footer-note">Severity and priority are automated estimates; crews verify on site.</span>
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <TopNav />
        <div className="app-main">
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
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
