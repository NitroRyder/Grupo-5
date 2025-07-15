# Grupo-5
PARA LA EJECUCIÓN DE ESTE CODIGO SEA LA MAS ADECUADA, ES NECESARIO TOMAR EN CONSIDERACIÓN LAS SIGUIENTES CONDICIONES:
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# A) ANTES DE EJECUTAR EL CODIGO ES NECESARIO REALIZAR LOS SIGUIENTES PASOS:
## -------------------------------------------------------------------------------------------------------------------------------
## 1°) CREAR BASE DE DATOS MYSQL UTILIZANDO EL SCRIPT QUE ESTÁ DENTRO DE LA CARPETA, ESPECEIFICAMENTE EL DOCUMENTO "Script_Mysql.txt"
## -------------------------------------------------------------------------------------------------------------------------------
## 2°) MODIFICAR ARCHIVO DEL CODIGO LLAMADO "database.py", TENIENDO EN CONSIFERACIÓN LO SIGUIENTE:

database = mysql.connector.connect(
    host="localhost", 				# EL CODIGO ESTÁ CREADO PARA EJECUTARSE DESDE: localhost
    user="root",      				# NOMBRE DEL USUARIO Mysql
    password="ZsefV1234$",			# CONTRASEÑA DEL PERFIL Mysql
    database="innovacion_y_emprendimiento_2"	# NOMBRE DE LA BASE DE DATOS(Especificada en documento "MySQL.txt")
)
## -------------------------------------------------------------------------------------------------------------------------------
## 3°) VERIFICAR QUE A LA HORA DE EJECUTAR EL CODIGO, LA TERMINAL SE ENCUENTRE DENTRO DE LA DIRECCIÓN:
  
   - [DIRECCIÓN]: PEP2_INOVACION\inovacion codigo-20250714T223246Z-1-001\inovacion codigo> 
   
   - ES MUY IMPORTANTE ESTE ULTIMO PASO, PUES ASÍ EL CODIGO ES CAPAZ DE UTILIZAR LA CARPETA "datos.json", DE LO CONTRARIO, NO
     FUNCIONARÁ EL CODIGO.
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# B) PARA FACILITAR EJECUCIÓN DEL CODIGO, SE HACE ENTREGA DE LAS LINEAS UTILIZADAS EN "postman" PARA PROBAR EL CODIGO:
## -------------------------------------------------------------------------------------------------------------------------------
## 1°) CREACIÓN DE USUARIO: 

[POST]: http://localhost:4000/credenciales

[JSON]: {"texto": "nombre: Jorge\napellido: Jimenez\nrut: 12.345.678-9"}

## -------------------------------------------------------------------------------------------------------------------------------
## 2°) INGRESO DE PREGUNTA AL SISTEMA:

[POST]: http://localhost:4000/ask

[JSON]: {"question": "Se atraso mi entrega"}

## -------------------------------------------------------------------------------------------------------------------------------
## 3°) AGREGADO DE INFORMACIÓN EXTA A PREGUNTA:

[POST]: http://localhost:4000/regravedad

[JSON]: {"informacion_extra": "No ha llegado desde hace 2 días"}

## -------------------------------------------------------------------------------------------------------------------------------
## 4°) PARA INGRESAR A LA INTERFAZ GRAFICA, ES NECESARIO VERIFICAR EN LA TERMINAL UNA VEZ SE EJECUTÓ EL CODIGO CORRECTAMENTE. 
    
   - HAY DOS OPCIONES PARA ACCEDER A ESTA:

    	a) LA DIRECCIÓN ES ENTREGADA DE LA FORMA http://

    	b) INGRESAR A: http://localhost:4000

||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
