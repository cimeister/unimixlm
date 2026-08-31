require('dotenv').config();
require('./helpers/passport')

const passport = require('passport');
const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const apiRouter = require('./routes/api');
const jwt = require('jsonwebtoken');
const cors = require('cors');

const app = express();

let corsOptions = {
  origin: ['https://drenchoman.github.io/blogapi_cms/', 'https://blog-client-zeta.vercel.app', 'https://drenchoman.github.io', 'http://localhost:3000'],
  optionsSuccessStatus: 200
}

const mongoDb = process.env.DB_URL;
mongoose.connect(mongoDb, {useUnifiedTopology: true, useNewUrlParser: true});
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'mongo connection error'));


app.use(bodyParser.urlencoded({extended: false}));
app.use(bodyParser.json());
app.options('*', cors(corsOptions));


app.use('/api', cors(corsOptions), apiRouter);


app.listen(process.env.PORT, () => console.log(`Server listening on ${process.env.PORT}`))