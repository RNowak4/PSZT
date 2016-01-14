import backend.Training;
import backend.TrainingImpl;
import com.googlecode.fannj.TrainingAlgorithm;

import com.googlecode.fannj.Fann;
import com.googlecode.fannj.Layer;
import com.sun.org.apache.xpath.internal.SourceTree;
import javafx.application.Application;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import backend.Training;
import backend.TrainingImpl;
import com.googlecode.fannj.TrainingAlgorithm;

public class Main extends Application {
    private Stage primaryStage;
    private BorderPane rootLayout;
    private BooleanProperty readyToWork = new SimpleBooleanProperty(false);

    @FXML
    private TextField directoryTF;
    @FXML
    private Button directoryButton;
    @FXML
    private Button processButton;
    @FXML
    private TextArea messageTextArea;
    @FXML
    private Button categorizeButton;
    @FXML
    private TextField resultTextField;
    @FXML
    private Label infoLabel;
    @FXML
    private MenuItem menuReset;
    @FXML
    private MenuItem menuClose;

    public static void main(String[] args) {
        System.setProperty("jna.library.path", "/usr/local/lib/");

        Fann fann = new Fann(layers);
        System.out.println(fann.getNumInputNeurons());
        System.out.println(fann.getNumOutputNeurons());
        System.out.println(fann.getTotalNumNeurons());

        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        this.primaryStage = primaryStage;
        this.primaryStage.setTitle("PSZT");

        initRootLayout();
        initGUIElements();
    }

    public void initRootLayout() {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(Main.class.getResource("MainWindow.fxml"));
            loader.setController(this);
            rootLayout = loader.load();

            Scene scene = new Scene(rootLayout);
            primaryStage.setScene(scene);
            primaryStage.show();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public void initGUIElements() {
        resultTextField.setEditable(false);
        messageTextArea.disableProperty().bind(readyToWork.not());
        categorizeButton.disableProperty().bind(readyToWork.not().or(messageTextArea.textProperty().isEmpty()));
        processButton.disableProperty().bind(directoryTF.textProperty().isEmpty());
        menuReset.disableProperty().bind(readyToWork.not());

        menuReset.setOnAction( event -> {
            //TODO
            //zresetowanie tego czego nauczyla sie siec neuronowa
            readyToWork.setValue(false);
        });

        menuClose.setOnAction( event -> {
            primaryStage.close();
        });

        directoryButton.setOnAction(event -> {
            System.out.println("event");
            DirectoryChooser chooser = new DirectoryChooser();
            File chosenDirectory = chooser.showDialog(rootLayout.getScene().getWindow());
            if (chosenDirectory != null) {
                directoryTF.setText(chosenDirectory.getAbsolutePath());
            }
        });

        processButton.setOnAction(event -> {
            infoLabel.setText("");
            File dir = new File(directoryTF.getText());
            if (!dir.exists()) {
                Alert alert = new Alert(Alert.AlertType.ERROR, "Type valid directory path!");
                alert.show();
            } else {
                //TODO
                //tutaj uruchomić uczenie
                //zmienna dir zawiera pliki xml do przetworzenia
                boolean result = true; // jesli operacja sie nie powiedzie ustawic na fail
                if(result) {
                    infoLabel.setTextFill(Paint.valueOf(Color.GREEN.toString()));
                    infoLabel.setText("PROCESSING SUCCEED");
                    readyToWork.setValue(true);
                } else {
                    infoLabel.setTextFill(Paint.valueOf(Color.RED.toString()));
                    infoLabel.setText("PROCESSING FAILED");
                }
            }
        });

        categorizeButton.setOnAction(event -> {
            String message = messageTextArea.getText();

            //TODO
            //tutaj wywołać funkcję zwracającą kategorię

            String result = "AAA"; //tu przypisać rezultat
            resultTextField.setText(result);
        });
    }
}
