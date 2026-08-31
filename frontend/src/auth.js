const TOKEN_KEY = "prism_admin_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export function isAuthenticated() {
  return Boolean(getToken());
}

export function getUsername() {
  const token = getToken();
  if (!token) return null;
  try {
    // Display only -- this decodes the JWT payload without verifying the signature,
    // which is fine here since it's never trusted for authorization, only shown in the UI.
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.sub || null;
  } catch {
    return null;
  }
}
