import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.speech.v1.RecognitionAudio;
import com.google.cloud.speech.v1.RecognitionConfig;
import com.google.cloud.speech.v1.RecognizeResponse;
import com.google.cloud.speech.v1.SpeechClient;
import com.google.cloud.speech.v1.SpeechRecognitionAlternative;
import com.google.cloud.speech.v1.SpeechRecognitionResult;
import com.google.cloud.speech.v1.SpeechSettings;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/changes")
public class TransController {

    @Value("${spring.cloud.gcp.speech.credentials.location}")
    private String credentialsPath;

    @PostMapping("/STT")
    public String transcribeAudio(@RequestParam("file") MultipartFile file) {
        try {
            SpeechSettings settings = SpeechSettings.newBuilder()
                    .setCredentialsProvider(FixedCredentialsProvider.create(
                            GoogleCredentials.fromStream(Files.newInputStream(Paths.get(credentialsPath)))
                    ))
                    .build();

            SpeechClient speechClient = SpeechClient.create(settings);
            byte[] audioBytes = file.getBytes();

            RecognitionAudio audio = RecognitionAudio.newBuilder()
                    .setContent(ByteString.copyFrom(audioBytes))
                    .build();

            RecognitionConfig config = RecognitionConfig.newBuilder()
                    .setEncoding(RecognitionConfig.AudioEncoding.LINEAR16)
                    .setSampleRateHertz(16000)
                    .setLanguageCode("ko-KR")
                    .build();

            RecognizeResponse response = speechClient.recognize(config, audio);

            StringBuilder transcript = new StringBuilder();
            for (SpeechRecognitionResult result : response.getResultsList()) {
                SpeechRecognitionAlternative alternative = result.getAlternativesList().get(0);
                transcript.append(alternative.getTranscript()).append("\n");
            }
            return transcript.toString();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @PostMapping("/OCR")
    public String detectTextFromImage(@RequestParam("file") MultipartFile file) {
        try {
            GoogleCredentials credentials = GoogleCredentials.fromStream(Files.newInputStream(Paths.get(credentialsPath)));

            ImageAnnotatorSettings settings = ImageAnnotatorSettings.newBuilder()
                    .setCredentialsProvider(FixedCredentialsProvider.create(credentials))
                    .build();
            ImageAnnotatorClient vision = ImageAnnotatorClient.create(settings);

            ByteString imgBytes = ByteString.copyFrom(file.getBytes());
            Image img = Image.newBuilder().setContent(imgBytes).build();

            Feature feature = Feature.newBuilder().setType(Feature.Type.TEXT_DETECTION).build();
            AnnotateImageRequest request = AnnotateImageRequest.newBuilder()
                    .addFeatures(feature)
                    .setImage(img)
                    .build();

            AnnotateImageResponse response = vision.batchAnnotateImages(Collections.singletonList(request)).getResponsesList().get(0);

            StringBuilder result = new StringBuilder();
            response.getTextAnnotationsList().forEach(annotation -> result.append(annotation.getDescription()).append("\n"));

            return result.toString();

        } catch (IOException e) {
            e.printStackTrace();
            return "Error occurred while processing the image.";
        }
    }
}
