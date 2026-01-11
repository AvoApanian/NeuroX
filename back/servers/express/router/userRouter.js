const express = require("express")

const {
    creationUser,
    loadInfo
} = require (
    "./../controller/userController"
)

const {authMiddleware} = require (
    "./../middleware/userMiddleware"
)

const userRouter = express.Router()

userRouter.get("/load",authMiddleware,loadInfo)
userRouter.post("/create",creationUser)

module.exports = userRouter