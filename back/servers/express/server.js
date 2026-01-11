const express = require("express");
const cors = require("cors"); 
require("dotenv").config();
const userRouter = require("./router/userRouter");
const server = express();
const PORT = process.env.PORT || 3001; // ← PORT en majuscule !

server.use(cors({
  origin: ["http://127.0.0.1:5173", "http://localhost:5173"],
  credentials: true
}));

server.use(express.json());
server.use("/user", userRouter);

server.listen(PORT, () => {
  console.log(`Server running on port -> ${PORT}`); // ← Parenthèses, pas backticks !
});