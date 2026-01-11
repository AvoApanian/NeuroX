import { useMutation } from "@tanstack/react-query";

export interface PostInter {
  url: string;
  obj: object;
  func: () => void;
}

export function usePostMethod({ url, func }: PostInter) {

  return useMutation({
    mutationFn: async (obj: object) => {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(obj),
      });
      if (!res.ok) throw new Error("Error -> POST");
      return res.json();
    },
    onSuccess: func
  });
}
