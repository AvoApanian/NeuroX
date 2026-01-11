import { createContext, useContext } from "react";
import TokenManager from "./token";

export const UserContext = createContext<TokenManager | null>(null);

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error("useUser must be used within UserContext.Provider");
  }
  return context;
};