import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FSDataOutputStream;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans_5i {
    public static int centroidCount = 0;
    public static List<ClusterCenter> centroids = new ArrayList<ClusterCenter>();
    public static List<ClusterCenter> previousCentroids = new ArrayList<ClusterCenter>();
    public static final double EARLY_CONVERGENCE_THRESHOLD = 0.001;

    public static class Point implements Writable {
        private double x, y;

        public Point() {
            x = 0.0;
            y = 0.0;
        }

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeDouble(x);
            dataOutput.writeDouble(y);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            x = dataInput.readDouble();
            y = dataInput.readDouble();
        }
    }

    public static class ClusterCenter implements Writable {
        public Point point = new Point(0, 0);
        public int id;
        public int count = 0;

        public List<Point> points;

        public ClusterCenter() {
            this.points = new ArrayList<Point>();
        }

        public ClusterCenter(int id, Point point) {
            this.points = new ArrayList<Point>();
            this.id = id;
            this.point = point;
        }

        public ClusterCenter(Point point) {
            this.point = point;
        }

        public Point getPoint() {
            return point;
        }

        public int get() {
            return count;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            point.write(dataOutput);
            dataOutput.writeInt(count);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            point.readFields(dataInput);
            count = dataInput.readInt();
        }

        public void addPoint(Point point) {
            points.add(point);
        }
    }

    public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Point> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            ClusterCenter nearestCenter = findNearestCluster(point);
            nearestCenter.addPoint(point);
            context.write(new IntWritable(nearestCenter.id), point);
        }

        private ClusterCenter findNearestCluster(Point point) {
            ClusterCenter nearestCenter = null;
            double temp = Double.MAX_VALUE;
            for (ClusterCenter center : centroids) {
                Point centerPoint = center.getPoint();
                double distance = calculateEuclideanDistance(point, centerPoint);
                if (distance < temp) {
                    nearestCenter = center;
                    temp = distance;
                }
            }
            return nearestCenter;
        }

        private double calculateEuclideanDistance(Point p1, Point p2) {
            double dx = p1.getX() - p2.getX();
            double dy = p1.getY() - p2.getY();
            return Math.sqrt(dx * dx + dy * dy);
        }
    }

    public static class KMeansCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY()));
            }
            ClusterCenter newCenter = computeNewCenter(points);
            context.write(key, newCenter.getPoint());
        }

        private ClusterCenter computeNewCenter(List<Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            for (Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            return new ClusterCenter(new Point(avgX, avgY));
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Point, IntWritable, Point> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY()));
            }
            ClusterCenter newCenter = computeNewCenter(points);
            context.write(key, newCenter.getPoint());
        }

        private ClusterCenter computeNewCenter(List<Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            for (Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            return new ClusterCenter(new Point(avgX, avgY));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMeans");

        String inputPath, kSeedData, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        kSeedData = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_2e_k7";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7;

        List<ClusterCenter> seedPoints = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(kSeedData))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                seedPoints.add(new ClusterCenter(j + 1, new Point(x, y)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        centroidCount = k;
        System.out.println("All seed points" + seedPoints);
        System.out.println("K value: " + centroidCount);

        List<ClusterCenter> selectedCentroids = getRandomCentroids(seedPoints, k);

        centroids = selectedCentroids;
        previousCentroids = new ArrayList<>(centroids);

        job.setJarByClass(KMeans_5i.class);
        job.setMapperClass(KMeans_5i.KMeansMapper.class);
        job.setCombinerClass(KMeansCombiner.class);
        job.setReducerClass(KMeans_5i.KMeansReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Point.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Point.class);


        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int maxIterations = 10;
        boolean converged = false;

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            boolean jobResult = job.waitForCompletion(true);
            if (!jobResult) {
                System.err.println("KMeans Job Failed");
                System.exit(1);
            }

            if (iteration < maxIterations) {
                previousCentroids.clear();
                previousCentroids.addAll(centroids);
                centroids.clear();

                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = new ArrayList<>();
                    for (ClusterCenter center : previousCentroids) {
                        if (center.id == i) {
                            clusterPoints.addAll(center.points);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        double totalX = 0.0;
                        double totalY = 0.0;
                        for (Point point : clusterPoints) {
                            totalX += point.getX();
                            totalY += point.getY();
                        }

                        double avgX = totalX / clusterPoints.size();
                        double avgY = totalY / clusterPoints.size();

                        centroids.add(new ClusterCenter(i, new Point(avgX, avgY)));
                    } else {
                        centroids.add(previousCentroids.get(i));
                    }
                }

                boolean hasConverged = true;
                for (int i = 0; i < k; i++) {
                    double distance = calculateEuclideanDistance(centroids.get(i).point, previousCentroids.get(i).point);
                    if (distance > EARLY_CONVERGENCE_THRESHOLD) {
                        hasConverged = false;
                        break;
                    }
                }

                if (hasConverged) {
                    System.out.println("Converged after " + iteration + " iterations.");
                    converged = true;
                    break;
                }
            }
        }

        if (converged) {

            System.out.println("Converged. Final Cluster Centers:");

            // Write convergence and final cluster center information to an output file
            Path clusterCenterOutputPath = new Path(outputPath, "cluster_centers.txt");
            try (FSDataOutputStream out = fs.create(clusterCenterOutputPath)) {

                String finalClusterInfo = "Converged. Final Cluster Centers:\n";
                out.write(finalClusterInfo.getBytes());

                for (ClusterCenter center : centroids) {
                    String outputLine = center.point.getX() + "," + center.point.getY() + "\n";
                    out.write(outputLine.getBytes());
                }
            }

            System.out.println("Final cluster centers and convergence information have been written to: " + clusterCenterOutputPath);
        }
    }


    private static List<ClusterCenter> getRandomCentroids(List<ClusterCenter> seedPoints, int k) {
        List<ClusterCenter> selectedCentroids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());

        if (k > seedPoints.size()) {
            k = seedPoints.size();
        }

        while (selectedCentroids.size() < k) {
            int randomIndex = rand.nextInt(seedPoints.size());
            ClusterCenter randomPoint = seedPoints.get(randomIndex);

            if (!selectedCentroids.contains(randomPoint)) {
                selectedCentroids.add(randomPoint);
            }
        }

        return selectedCentroids;
    }

    private static double calculateEuclideanDistance(Point p1, Point p2) {
        double dx = p1.getX() - p2.getX();
        double dy = p1.getY() - p2.getY();
        return Math.sqrt(dx * dx + dy * dy);
    }
}
