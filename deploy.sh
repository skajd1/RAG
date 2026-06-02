#!/bin/bash

# 1. 프로젝트 경로 이동
cd ~/metsabrain

# 2. 최신 코드 가져오기
echo "🔄 Updating source code from Git..."
git pull origin main

# 3. 특정 앱 서비스만 중지 및 삭제 (인프라 무중단)
echo "🧹 Stopping and removing app containers..."
docker compose stop backend-api frontend-ui
docker compose rm -f backend-api frontend-ui

# 4. 앱 서비스 재빌드 및 재시작
echo "🏗️ Rebuilding and starting app services..."
docker compose --env-file .env up -d --build backend-api frontend-ui

# 5. 결과 확인
echo "✅ Deployment successful!"
docker ps | grep -E "backend-api|frontend-ui"
