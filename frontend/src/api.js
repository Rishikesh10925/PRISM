import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

const client = axios.create({ baseURL: API_BASE });

export async function submitReport(imageFile, latitude, longitude) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("latitude", latitude);
  form.append("longitude", longitude);
  const { data } = await client.post("/report", form);
  return data;
}

export async function fetchPriorityList(status) {
  const { data } = await client.get("/priority-list", { params: status ? { status } : {} });
  return data;
}

export async function updateStatus(detectionId, status) {
  const { data } = await client.patch(`/potholes/${detectionId}/status`, { status });
  return data;
}

export function pdfReportUrl() {
  return `${API_BASE}/report/pdf`;
}
