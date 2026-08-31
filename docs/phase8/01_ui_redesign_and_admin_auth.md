# Phase 8 / Task 1 (extended) — Professional UI Redesign & Admin Authentication

Two related changes on top of the earlier dashboard polish (loading states, hover feedback):
a full visual redesign, and gating the admin dashboard behind a real login instead of leaving
it open to anyone who knows the `/admin` URL.

## Admin authentication

Previously `/admin` and every worklist-management endpoint (`GET /api/potholes`,
`GET /api/priority-list`, `PATCH /api/potholes/{id}/status`, `GET /api/report/pdf`) were
completely open — anyone with the URL could see and edit the repair worklist. Citizen
report submission (`POST /api/report`) is deliberately still open; the fix only gates the
*admin* side.

- **Backend** ([app/auth.py](../../backend/app/auth.py),
  [app/routers/auth.py](../../backend/app/routers/auth.py)): `POST /api/auth/login` checks a
  username/password against `ADMIN_USERNAME` / `ADMIN_PASSWORD_HASH` (bcrypt, via `passlib`)
  and returns a signed JWT (`pyjwt`, 8-hour expiry by default). Every admin-only route now
  depends on `get_current_admin`, which verifies the bearer token and rejects missing/invalid/
  expired ones with 401. Verified by 4 new tests in `backend/tests/test_api.py`
  (`test_login_succeeds_with_correct_credentials`, two wrong-credential cases, and
  `test_admin_endpoints_reject_missing_or_bad_token`), plus every existing protected-endpoint
  test updated to send a real token — 12/12 backend tests pass.
- **Default credentials** (local dev only, see `backend/.env.example`): username `admin`,
  password `PrismAdmin#2026`. **Change `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, and
  `JWT_SECRET` via real environment variables before this is ever exposed beyond localhost** —
  the defaults baked into `config.py` are explicitly documented as insecure placeholders, not a
  real credential.
- **Frontend** (`frontend/src/auth.js`, `pages/AdminLogin.jsx`,
  `components/ProtectedRoute.jsx`): token stored in `localStorage`, attached to every API
  request via an axios interceptor, cleared automatically on any 401 response. `/admin` is
  wrapped in `ProtectedRoute`, which redirects to `/admin/login` when there's no token;
  logging in redirects back to wherever the admin was headed. The PDF report link changed
  from a plain `<a href>` (which can't carry an Authorization header) to a button that fetches
  the PDF as an authenticated blob and triggers the download client-side.
- Verified end-to-end through a real browser: unauthenticated `/admin` redirects to login;
  wrong password shows an error and doesn't authenticate; correct password logs in and lands
  on the dashboard; "Sign out" clears the session and redirects back to login; the PDF button
  downloads a real PDF (`application/pdf`, non-trivial size) only while authenticated.

## Visual redesign

Replaced the ad-hoc styling from the earlier polish pass with an actual design system in
`frontend/src/index.css`: CSS custom properties for color/spacing/radius/shadow, the Inter
typeface (Google Fonts, loaded in `index.html`), and `color-mix()`-based tinting so severity/
priority colors (already defined in `ScoreBadge.jsx`, unchanged) automatically produce
matching tinted backgrounds without hand-picking a second palette.

- **Citizen upload page**: hero header with an icon badge, a real drag-and-drop dropzone (not
  just click-to-browse — `onDragOver`/`onDrop` wired to the same file-handling path as the file
  input), and the result screen rebuilt around two large "score tiles" (big number + category,
  color-coded) replacing the small text-only badges, for much stronger visual hierarchy on the
  two numbers that matter most.
- **Admin dashboard**: the old black bar was replaced by a proper toolbar — page title + live
  stat pills on the left, a user identity pill (avatar initial + username + "Sign out") on the
  right. Sidebar, map, and table keep their Phase-8-polish behavior (loading skeletons, hover
  states, sticky header) with refined spacing, borders, and shadows to match the new system.
- **Login page**: a centered card on a subtle branded background, consistent with the rest of
  the app rather than a bare unstyled form.

Confirmed via `npm run build` (no errors) and a full real-browser pass (login flow, citizen
upload through to a real detection result, PDF download) with zero console errors and zero
failed requests.
