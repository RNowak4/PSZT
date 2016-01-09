import com.googlecode.fannj.Fann;
import com.googlecode.fannj.Layer;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        System.setProperty("jna.library.path", "./lib/");

        List<Layer> layers = new ArrayList<>();
        layers.add(Layer.create(3000));
        layers.add(Layer.create(300));
        layers.add(Layer.create(30));
        layers.add(Layer.create(3));

        Fann fann = new Fann(layers);
        System.out.println(fann.getNumInputNeurons());
        System.out.println(fann.getNumOutputNeurons());
        System.out.println(fann.getTotalNumNeurons());
    }
}
