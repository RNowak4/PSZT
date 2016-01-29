import backend.TextAnalyzer;
import backend.TextAnalyzerImpl;
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
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.stream.Collectors;

public class Main extends Application {
    private static final double DEFAULT_ERROR = 0.01;
    private static final int DEFAULT_MAX_EPOCHS = 50;
    private static final Class DEFAULT_LEARNING = ResilientPropagation.class;
    private TextAnalyzer textLearner = new TextAnalyzerImpl(DEFAULT_ERROR, DEFAULT_MAX_EPOCHS, DEFAULT_LEARNING);
    private Stage primaryStage;
    private BorderPane rootLayout;
    private BooleanProperty readyToWork = new SimpleBooleanProperty(false);

    @FXML
    private TextField directory1TF;
    @FXML
    private Button directory1Button;
    @FXML
    private Button process1Button;
    @FXML
    private TextField directory2TF;
    @FXML
    private Button directory2Button;
    @FXML
    private Button process2Button;
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
//        launch(args);

        // TODO reczne uczenie z pliku - bez gui
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
        process2Button.disableProperty().bind(directory2TF.textProperty().isEmpty());
        menuReset.disableProperty().bind(readyToWork.not());

        menuReset.setOnAction(event -> {

            readyToWork.setValue(false);
        });

        menuClose.setOnAction(event -> {
            primaryStage.close();
        });

        directory1TF.setEditable(false);
        directory2TF.setEditable(false);
        directory2Button.setDisable(true);

        directory1Button.setOnAction(event1 -> {
            FileChooser chooser = new FileChooser();
            File chosenFile = chooser.showOpenDialog(rootLayout.getScene().getWindow());
            if (chosenFile != null) {
                directory1TF.setText(chosenFile.getAbsolutePath());
            }
        });

        process1Button.setOnAction(event1 -> {
            File file = new File(directory1TF.getText());
            if (!file.exists()) {
                Alert alert = new Alert(Alert.AlertType.ERROR, "Type valid file path!");
                alert.show();
            } else {
                textLearner.loadCategoriesFromFile(directory1TF.getText());
                directory2Button.setDisable(false);
            }
        });

        directory2Button.setOnAction(event -> {
            FileChooser chooser = new FileChooser();
            File chosenFile = chooser.showOpenDialog(rootLayout.getScene().getWindow());
            if (chosenFile != null) {
                directory2TF.setText(chosenFile.getAbsolutePath());
            }
        });

        process2Button.setOnAction(event -> {
            File file = new File(directory2TF.getText());
            if (!file.exists()) {
                Alert alert = new Alert(Alert.AlertType.ERROR, "Type valid file path!");
                alert.show();
            } else {
                boolean result = textLearner.trainFile(directory2TF.getText());
                if (result) {
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

            MLData data = textLearner.analyzeText(message);
            int position = 0;
            for (int i = 0; i < data.getData().length; i++) {
                if (data.getData()[i] > data.getData()[position]) {
                    position = i;
                }
            }

            final Integer pos = new Integer(position);

            if (data.getData().length != textLearner.getCategories().size()) {
                System.out.println("ERROR!!!!");
            }


            String result = textLearner.getCategories().entrySet()
                    .stream()
                    .filter(entry -> entry.getValue().equals(pos))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList()).get(0);
            resultTextField.setText(result);
        });
    }
}