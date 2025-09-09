# Generador de mallas 3D a partir de dibujos de los perfiles de un objeto

Genera automáticamente una malla 3D cerrada a partir de tres imágenes 2D ortogonales de un objeto (alzado, planta y perfil). Incluye una interfaz gráfica para cargar vistas, ajustar parámetros y exportar a OBJ.

## Instalación
pip install -r requirements.txt

## Guía de uso
1. Ejecuta la interfaz: python src/gui.py
2. Carga las imágenes de las vistas (con los botones o arrastrando y soltando los archivos en la ventana)
3. Ajusta los parámetros y tolerancias para mejorar la reconstrucción
4. Pulsa "Reconstruir modelo"
5. Exporta el resultado con "Exportar" (formato OBJ)

