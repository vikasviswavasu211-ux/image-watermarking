import app from '../server';

// Vercel Serverless Function config:
// - bodyParser: false  → lets Express + multer handle multipart/form-data uploads
// - responseLimit: 50mb → allows large base64 image responses
// - externalResolver: true → prevents Vercel from warning about unresolved requests
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
    externalResolver: true,
  },
};

export default app;
