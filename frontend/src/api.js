import axios from "axios";
import { clearToken, getToken } from "./auth";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

const client = axios.create({ baseURL: API_BASE });

client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken();
    }
    return Promise.reject(error);
  }
);

export async function submitReport(imageFile, latitude, longitude) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("latitude", latitude);
  form.append("longitude", longitude);
  const { data } = await client.post("/report", form);
  return data;
}

export async function login(username, password) {
  const { data } = await client.post("/auth/login", { username, password });
  return data.access_token;
}

export async function fetchPriorityList(status) {
  const { data } = await client.get("/priority-list", { params: status ? { status } : {} });
  return data;
}

export async function updateStatus(detectionId, status) {
  const { data } = await client.patch(`/potholes/${detectionId}/status`, { status });
  return data;
}

export async function downloadPdfReport() {
  const response = await client.get("/report/pdf", { responseType: "blob" });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = "prism_worklist.pdf";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
