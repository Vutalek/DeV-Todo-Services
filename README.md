# DeV-Todo-Services

Проверка главного сервиса 
```
curl.exe -X POST -H "Content-Type: application/json" -d "@app/check.json" http://ip:port/app/v1/send
```

Проверка mcp сервиса
```
curl.exe -X POST -H "Content-Type: application/json" -d "@mcp/check.json" http://ip:port/mcp/v1/sendtask
```