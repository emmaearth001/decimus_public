"use client";

interface Props {
  message: string;
  type: "info" | "error" | "success";
}

export default function StatusBar({ message, type }: Props) {
  if (!message) return null;
  const color = type === "error" ? "text-danger" : type === "success" ? "text-success" : "text-dim";
  return <div className={`text-xs ${color} py-2`}>{message}</div>;
}
