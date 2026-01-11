import { useMutation } from "@tanstack/react-query";
import { useUser } from "../Context";

export const formDataPost = (url: string) => {
  const tokenManager = useUser();
  
  return useMutation({
    mutationFn: async (formData: FormData) => {
      console.log("Token:", tokenManager.token);
      console.log("Token type:", typeof tokenManager.token);
      
      if (!tokenManager.token) {
        throw new Error("No token available");
      }

      const headers = {
        'Authorization': `Bearer ${tokenManager.token}`,
      };
      
      console.log("Headers being sent:", headers);

      const response = await fetch(url, {
        method: "POST",
        headers: headers,
        body: formData,
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        const error = await response.json();
        console.error("Error response:", error);
        throw new Error(error.detail || "Error -> formData");
      }

      return response.json();
    },
  });
};