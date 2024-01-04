from pyspark.sql import SparkSession
import multiprocessing

def square_number(x):
    return x*x

if __name__ == '__main__':
    # Create or retrieve a Spark session
    spark = SparkSession.builder.appName("SquareNumbers").getOrCreate()

    # Get the number of CPUs available on the system
    num_cpus = multiprocessing.cpu_count()

    # Deduct one CPU for other processes and set the parallelism
    spark.conf.set("spark.default.parallelism", num_cpus - 1)

    # List of numbers
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Convert the list to an RDD
    numbers_rdd = spark.sparkContext.parallelize(numbers)

    # Map each number to its square
    squared_numbers = numbers_rdd.map(square_number).collect()

    # Print the squared numbers
    print(squared_numbers)

    # Stop the Spark session
    spark.stop()