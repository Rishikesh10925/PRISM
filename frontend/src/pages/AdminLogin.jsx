import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { login } from "../api";
import { setToken } from "../auth";
import Spinner from "../components/Spinner";
import { PrismMark } from "../components/Logo";
import { AlertCircleIcon, EyeIcon, EyeOffIcon, LockIcon } from "../components/Icons";

export default function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
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
    <main className="auth-shell">
      <div className="auth-column">
        <form className="auth-card" onSubmit={handleSubmit}>
          <div className="auth-brand">
            <PrismMark size={40} />
            <div>
              <div className="auth-brand-word">PRISM</div>
              <div className="auth-brand-sub">Repair console</div>
            </div>
          </div>

          <h1 className="auth-title">Sign in</h1>
          <p className="auth-subtitle">Municipal staff access to the pothole repair worklist.</p>

          <div className="field">
            <label htmlFor="login-username">Username</label>
            <input
              id="login-username"
              className="input"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </div>

          <div className="field">
            <label htmlFor="login-password">Password</label>
            <div className="input-affix">
              <input
                id="login-password"
                className="input"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                required
              />
              <button
                type="button"
                className="input-affix-btn"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                aria-pressed={showPassword}
              >
                {showPassword ? <EyeOffIcon size={18} /> : <EyeIcon size={18} />}
              </button>
            </div>
          </div>

          <div aria-live="polite">
            {error && (
              <p className="alert alert-error" role="alert">
                <AlertCircleIcon size={18} />
                <span>{error}</span>
              </p>
            )}
          </div>

          <button type="submit" className="btn btn-primary btn-block btn-lg" disabled={submitting}>
            {submitting ? (
              <>
                <Spinner size="sm" /> Signing in…
              </>
            ) : (
              <>
                <LockIcon size={16} /> Sign in
              </>
            )}
          </button>
        </form>
        <p className="auth-foot">
          Reporting a pothole? <Link to="/">Go to the public reporting page</Link>
        </p>
      </div>
    </main>
  );
}
