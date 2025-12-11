#Dog2Human - Final Report

#Summary

Dog2Human is a web application that turns pictures of dogs into their human counter parts. At the current state of the project the web application does not work because I am going over the memory limit on Render. However, the local version works as intended and even though my model is not generating perfect humans but better put from my friends "unfathomable horrors" I think its come a long way. The system uses PostgreSQL for the database via Supabase to store users and generations, FastAPI for the backend, Tailwind CSS-styled HTML templates for the frontend, and a UNet based generator trained on dog and human image pairs. For demo I ran the full GAN model locally with GPU acceleration, and host a lightweight version of my app on render to show real authentication, SQL read and writes, and the live site. The link to the website is bellow. 

#Link to site (do not expect working results)
https://dog2human-mgmo.onrender.com/



#Demo
## Demo

[Watch the demo video](Dog2HumanDemo1.mp4)


#System Design

### Data Model

**PostgreSQL** database tables: 

- **users**
  - `id` (PK)
  - `email` (unique)
  - `password_hash` (bcrypt via Passlib)
  - `created_at`

- **dog_images**
  - `id` (PK)
  - `user_id` (FK → users.id)
  - `file_path` (relative path under `/static/uploads/...`)
  - `created_at`

- **generated_images**
  - `id` (PK)
  - `user_id` (FK → users.id)
  - `dog_image_id` (FK → dog_images.id)
  - `file_path` (relative path under `/static/generated/...`)
  - `model_version` (e.g. `"gan_epoch_20"`)
  - `created_at`


### Architecture

1. **Client**
   - Uses HTML templates and Tailwind.
   - Uploads an image using a form on `/app`.
   - Stores a JWT token in `localStorage` after login.
   - Calls the `/generate` endpoint with `Authorization: Bearer <token>`.

2. **FastAPI backend**
   - Endpoints:
     - `POST /auth/register` – register user, hash password, store in Postgres.
     - `POST /auth/login` – authenticate user, return JWT.
     - `GET /app` – app UI.
     - `POST /generate` – protected endpoint that
       - Saves uploaded dog image to `static/uploads/`
       - Runs the UNet GAN generator for local
       - Saves generated human image to `static/generated/`
       - Inserts rows into `dog_images` and `generated_images`.
       - Returns JSON with URLs of uploaded and generated images.
     - `GET /gallery` – returns a page with recent generations, up to 50.

3. **PostgreSQL**
   - Stores persistent users and generation metadata like time generated.
   - Accessed using a `DATABASE_URL` stored in an environment variable.

4. **Model (UNetDog2Human)**
   - Implemented in `src/models.py`.
   - Trained offline with a GAN training loop and perceptual loss.
   - Weights saved as `gan_epoch_20.pt`.
   - In the local demo:
     - I load the checkpoint, run inference on GPU, and show high quality outputs.


## What I learned

1. It was fun to learn how to make a full stack application when there are no guidlines on what to use. It's also nice to now know about all the tools to help you make an application quickly like fastAPI and Supabase.
2. I learned how to use relational databases in a program that I wrote. I have done this before but this time felt more independant. Took me a second how to manage sessions and debug tables. Also ran into a lot of connection issues when trying to deploy my application.
3. I learned how useful having a GPU can be. I have a pc I built and the GPU in it came in handy. I also learned that my code can crash on a free 512 MiB container. Hence why my web app does not work. 
4. I implemented user registration, bcrypt password hashing, JWT authentication, and routes in FastAPI. 
5. Some of the biggest issues I ran into were managing dependencies, handling environment variables, and IPv3 vs IPv6 issues and getting Supabase to work with Render. 

## Does it integrate with AI

Yes! The core of the project is an image to image generator that maps dog faces to human faces. 

I trained a **UNet-based generator** as part of a GAN setup, using dog and human image pairs.
- The training loop included:
  - An adversarial loss, the generator trying to feel a discriminator.
  - A perceptual loss to preserve high level structure.

## How did I use AI 

I used AI to help me with a lot of the debugging. It helped me figure out how to use CUDA instead of my CPU. It helped me figure out issues with my .env handling. It also helped me with the FastAPI routes and the password hashing. My original HTML was really ugly and I had it help me restyle my website. I also had a lot of trouble deploying and AI helped me understand the errors from Render logs. Using AI helped a lot for understanding what was going on, especially becasue I have not used some of these softwares before. 

## Why is this project interesting to me

I am taking a deep learning class and we learned about GAN diffusion models. I thought they were cool and I wanted to try implementing one on my own. I also just really like dogs and I thought the project idea sounded fun, who doesn't want to see their beloved dog as a human? I also think that this project has some personality and it's technical and relevant enough to put on a resume. 

## Key learnings
1. A learned there is a big difference from something working on my machine rather than working as a full application. Getting PyTorch models to run on my constrained environment was a lot easier than getting them to run in a production environment. 
2. Having your environment and configurations set up correctly are as important as code. Having your important URL's and keys configured incorrectly will stop your full stack from working. 
3. Having good table design is really important for having a clear and easy to manage application. Having seperate tables for users dog_images and generated_images made it easy to query all generations from one user and makes the whole application more scalable.

##Explanations

For persistance and authentication I used Supabase hosted PostgreSQL database with SQL, bcrypt hashed passwords, and JWT based login to protect endpoints. The FastAPI runs on a single render instance now but he design is easy to scale horizontally. Performance is limited by ML inference its fast on my GPU but is constrained on the small cloud I tried using. The Database reads and writes are lightweight. FastAPI and Uvicorn together handle concurrent requests with session scoping. For failover and avaliability I rely on the managed uptime and backups from Render and Supabase. 

## Link to post in class channel

https://teams.microsoft.com/l/message/19:aa68ccc0d80e460bb785b2b9085c26ba@thread.tacv2/1765416950267?tenantId=c6fc6e9b-51fb-48a8-b779-9ee564b40413&groupId=89222c8a-2991-4873-8d7d-10b1cacaebf4&parentMessageId=1765416950267&teamName=CS%20452%20001%20(Fall%202025)&channelName=Report%20-%20Final%20Project&createdTime=1765416950267


## no, don't share (my application doesn't work in its web app form so I don't think its worth it)
