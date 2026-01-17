import { GetMethod } from "./fetcher/get";

export default class TokenManager {
  public token: string | null = null;
  private Apiurl = import.meta.env.VITE_backExpress;

  constructor() {
    this.loadTokenFromStorage();
  }

  private loadTokenFromStorage() {
    this.token = localStorage.getItem("userToken");
    
    if (this.token) {
      if (this.isTokenExpired(this.token)) {
        console.log("Token expired, removing it");
        this.removeToken();
      } else {
        console.log("Valid token found in storage");
      }
    } else {
      console.log("No token found in storage");
    }
  }

  private isTokenExpired(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const expiry = payload.exp * 1000;
      return Date.now() >= expiry;
    } catch (error) {
      console.error("Error parsing token:", error);
      return true;
    }
  }

  private removeToken() {
    localStorage.removeItem("userToken");
    this.token = null;
  }

  private saveToken(token: string) {
    localStorage.setItem("userToken", token);
    this.token = token;
  }

  public hasToken(): boolean {
    return this.token !== null;
  }

  public async loadData() {
    if (!this.token) {
      throw new Error("No token available");
    }
    
    try {
      return await GetMethod({
        url: `${this.Apiurl}/user/load`,
        headers: {
          Authorization: `Bearer ${this.token}`,
        },
      });
    } catch (error: any) {
      if (error.status === 401 || error.status === 404) {
        console.log("Token invalid or user not found, removing token");
        this.removeToken();
      }
      throw error;
    }
  }

  public async createUser() {
    if (this.token != null) {
      console.log("Token already exists, skipping creation");
      return;
    }

    console.log("Creating new user...");
    const res = await fetch(`${this.Apiurl}/user/create`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
    });

    if (!res.ok) throw new Error("Error creating user");
    
    const data = await res.json();
    this.saveToken(data.token);
    
    console.log("User created successfully");
    return data;
  }
}