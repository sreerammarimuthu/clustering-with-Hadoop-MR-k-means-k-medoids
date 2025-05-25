import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans_SI {
    public static int centroidCount = 0;
    public static List<ClusterCenter> centroids = new ArrayList<ClusterCenter>();

    public static List<Point> k_seeds = new ArrayList<Point>();

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

    public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private Path tempOutputPath;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");

            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);

            Point point = new Point(x, y);

            ClusterCenter nearestCenter = findNearestCluster(point);
            nearestCenter.addPoint(point);

            context.write(new IntWritable(nearestCenter.id), new Text(x + ":" + y));
        }

        private ClusterCenter findNearestCluster(Point point) {
            ClusterCenter nearestCenter = null;
            double temp = Double.MAX_VALUE;

            for (ClusterCenter center : centroids) {
                Point centerPoint = center.getPoint();

                double distance = calculateEuclideanDistance(point, centerPoint);

                if (distance < temp) {
                    nearestCenter = center;
                }
                temp = distance;
            }

            return nearestCenter;
        }

        private double calculateEuclideanDistance(Point p1, Point p2) {
            double dx = p1.getX() - p2.getX();
            double dy = p1.getY() - p2.getY();
            return Math.sqrt(dx * dx + dy * dy);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        private List<ClusterCenter> clusterCenters = new ArrayList<>();
        private List<Point> k_seed_updated = new ArrayList<Point>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // Load cluster centers from a file
            // You can implement this based on your data input format
        }

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Text value : values) {
                String[] parts = value.toString().split(":");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                points.add(new Point(x, y));
            }

            ClusterCenter newCenter = computeNewCenter(points);

            context.write(key, new Text(newCenter.point.getX() + "," + newCenter.point.getY()));
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

        String inputPath, k_seed_data, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        k_seed_data = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_2a_k333";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 3; // Set the desired value of k

        // Read all points from the seed points file and store them in a list
        List<ClusterCenter> seedPoints = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(k_seed_data))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) { // Read the first k lines
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                seedPoints.add(new ClusterCenter(j+1, new Point(x, y)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set the centroidCount to k
        centroidCount = k;

        List<ClusterCenter> selectedCentroids = getRandomCentroids(seedPoints, k);

        // Initialize your centroids list with the selected random k points
        centroids = selectedCentroids;

        job.setJarByClass(KMeans_SI.class);
        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int r = 1;

        for (int i = 0; i < r; i++) {
            job.waitForCompletion(true);
        }
    }

    private static List<ClusterCenter> getRandomCentroids(List<ClusterCenter> seedPoints, int k) {
        List<ClusterCenter> selectedCentroids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis()); // Use current time as the seed

        // Ensure k is not greater than the number of available seed points
        if (k > seedPoints.size()) {
            k = seedPoints.size();
        }

        while (selectedCentroids.size() < k) {
            int randomIndex = rand.nextInt(seedPoints.size());
            ClusterCenter randomPoint = seedPoints.get(randomIndex);

            // Avoid selecting the same point multiple times
            if (!selectedCentroids.contains(randomPoint)) {
                selectedCentroids.add(randomPoint);
            }
        }

        return selectedCentroids;
    }
}
