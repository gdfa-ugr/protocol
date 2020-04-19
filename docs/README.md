Wiki para documentación de la herramienta
===============

Se describen las reglas básicas a emplear para documentar las funciones de la herramienta.

## Descripción de la función

La descripción debe indicar lo que hace la función, pero NO cómo lo hace. Se ha de ser lo más breve y preciso posible.

Se muestra a continuación un ejemplo de la función “river_flow”.

```
def river_flow(file_name, path='.', null_values=(-99.9, -99.99, -999, -9999, 990), name='values'):
    """Read flow data file

    Args:
        file_name (str): file name
        path (str, optional): path
        null_values (list, optional): considered null values
        name (str, optional): name of data variable

    Returns:
        pd.Series: flow data
    """
    data = pd.read_csv(os.path.join(path, file_name), usecols=[1, 2], parse_dates={'datetime': [0]},
                       index_col=0, na_values=null_values, squeeze=True)
    data.rename(name, inplace=True)

    return data
```

## Tipos de datos de los argumentos de entrada y salida

Los parámetros de entrada se especificarán con el nombre de la variable y entre paréntesis el tipo de dato seguido de una descripción. Los parámetros de salida se especificarán con el tipo de dato seguido de la descripción, sin indicar su nombre. Los tipos de datos más habituales son:

- **int**: tipo de dato entero
- **float**: tipo de dato flotante
- **str**: tipo de dato cadena o *string*. No utilizar *unicode* puesto que así se establece en Python 3.
- **tuple**: tipo de dato tupla
- **list**: tipo de dato lista
- **list-like**: tipos de datos compatible con lista: lista, tupla...
- **bool**: tipo de dato que contiene *True* o *False*
- **np.array**: tipo de dato *array* de NumPy
- **pd.DataFrame** y **pd.Series**: tipos de datos DataFrame y Series (ojo a las mayúsculas) de Pandas
- **pd.Timedelta**: tipo de dato periodo de tiempo del módulo Pandas
- **datetime**: tipo de datos del módulo estándar de Python *datetime*
- **np.datetime64**: tipo de datos del módulo NumPy


## Parámetro opcional

Cuando un parámetro de la función sea opcional se deberá indicar con coma seguido de *optional*. Ver ejemplo anterior.

## Un parámetro puede tomar varios tipos de dato

Cuando un parámetro pueda tomar varios tipos de datos se deberá indicar con la palabra or, como se muestra en el siguiente ejemplo:

```
def is_positive(value=1):
    """Checks if a number is positive or not

    Args:
        value (int or float, optional): number to check

    Returns:
        boolean: True if value is positive, and False otherwise
    """ 
```

## Varios parámetros de salida

Cuando la función devuelva varios parámetros de salida se deberá indicar del siguiente modo:

```
def simar_header(file_name):
    """Detecting if file has header. If positive, extract number of lines and identifier code

    Args:
        file_name (str): file name

    Returns:
        tuple:

            - (int): number of header lines
            - (int): data header line
            - (str): identifier of SIMAR file
    """
```

**IMPORTANTE**: Dejar una línea en blanco antes más la indentación para que reconozca correctamente la lista.

## Un parámetro puede tomar un conjunto de valores definidos

Cuando un determinado parámetros pueda tomar un conjunto de valores definidos (y únicamente esos valores) se deberá especificar en la documentación del siguiente modo:

```
def simar(file_name, path='.', null_values=(-99.9, -99.99, -999, -9999, 990), columns='std'):
    """Read SIMAR file

    Args:
        file_name (str): file name
        path (str, optional): path
        null_values (list, optional): considered null values
        columns (str, optional):

            - std: hs (significant wave height), tp (peak period), dh (mean wave direction),
              vv (wind velocity) and dv (mean wind direction)
            - all: all the variables

    Returns:
        tuple:

            - (pd.DataFrame): climate agents variables
            - (str): identifier of SIMAR file
    """
```

**IMPORTANTE**: Dejar una línea en blanco antes más la indentación para que reconozca correctamente la lista.

