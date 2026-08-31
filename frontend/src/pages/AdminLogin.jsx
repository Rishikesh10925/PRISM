import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { login } from "../api";
import { setToken } from "../auth";
import Spinner from "../components/Spinner";

export default function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const token = await login(username, password);
      setToken(token);
      const redirectTo = location.state?.from || "/admin";
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(
        err.response?.status === 401
          ? "Incorrect username or password."
          : "Could not sign in. Is the backend running?"
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <form className="auth-card" onSubmit={handleSubmit}>
        <div className="auth-logo">🛣️</div>
        <h1>Admin sign in</h1>
        <p className="subtitle">Sign in to manage the repair worklist.</p>

        <label className="field">
          <span>Username</span>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        {error && <p className="error-text">{error}</p>}

        <button type="submit" className="primary-button" disabled={submitting}>
          {submitting ? (
            <>
              <Spinner size="sm" /> Signing in...
            </>
          ) : (
            "Sign in"
          )}
        </button>
      </form>
    </div>
  );
}
