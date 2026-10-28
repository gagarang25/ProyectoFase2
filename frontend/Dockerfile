# Frontend del chatbot: archivos estáticos (HTML, CSS, JS) servidos por nginx.
# nginx también funciona como proxy inverso: las llamadas a /api, /docs y /openapi.json
# se reenvían al contenedor del backend, así el navegador habla con un solo origen.
FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/public /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost:8080/ >/dev/null || exit 1
