import com.google.api.gax.core.FixedCredentialsProvider;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.speech.v1.RecognitionAudio;
import com.google.cloud.speech.v1.RecognitionConfig;
import com.google.cloud.speech.v1.RecognizeResponse;
import com.google.cloud.speech.v1.SpeechClient;
import com.google.cloud.speech.v1.SpeechRecognitionAlternative;
import com.google.cloud.speech.v1.SpeechRecognitionResult;
import com.google.cloud.speech.v1.SpeechSettings;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BoundingPoly;
import com.google.cloud.vision.v1.EntityAnnotation;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.Image;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageAnnotatorSettings;
import com.google.cloud.vision.v1.Vertex;
import com.google.protobuf.ByteString;
import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
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

    private static final Set<String> SPECIAL_TERMS = new HashSet<>(Arrays.asList("호선", "출입구", "지하철"));

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

            // 텍스트 블록 추출
            List<EntityAnnotation> textAnnotations = response.getTextAnnotationsList();
            if (textAnnotations.isEmpty()) {
                return "No text found.";
            }

            // 텍스트 블록을 면적 기준으로 정렬
            List<EntityAnnotation> sortedTextAnnotations = textAnnotations.stream()
                    .skip(1) // 첫 번째는 전체 텍스트 블록이므로 스킵
                    .sorted(Comparator.comparingDouble(this::calculateBoundingBoxArea).reversed())
                    .collect(Collectors.toList());

            // 가장 큰 텍스트와 두 번째로 큰 텍스트의 면적
            double largestArea = sortedTextAnnotations.stream().findFirst().map(this::calculateBoundingBoxArea).orElse(0.0);
            double secondLargestArea = sortedTextAnnotations.stream().skip(1).findFirst().map(this::calculateBoundingBoxArea).orElse(0.0);

            // 조건에 맞는 단어를 포함하는 경우 한글과 숫자만 추출
            boolean specialCondition = textAnnotations.stream()
                    .map(EntityAnnotation::getDescription)
                    .anyMatch(text -> SPECIAL_TERMS.contains(text) || text.endsWith("역"));

            List<String> finalText;
            if (specialCondition) {
                finalText = textAnnotations.stream()
                        .skip(1) // 첫 번째는 전체 텍스트 블록이므로 스킵
                        .map(EntityAnnotation::getDescription)
                        .map(this::extractHangulAndNumbers)
                        .filter(text -> !text.isEmpty())
                        .collect(Collectors.toList());
            } else {
                // 면적 차이를 확인하여 출력 결정
                List<EntityAnnotation> selectedAnnotations;
                if (secondLargestArea > 0 && largestArea / secondLargestArea > 2) {
                    // 면적 차이가 두 배 이상 나면 가장 큰 것만 선택
                    selectedAnnotations = Collections.singletonList(sortedTextAnnotations.get(0));
                } else {
                    // 그렇지 않으면 가장 큰 것과 두 번째로 큰 것 선택
                    selectedAnnotations = sortedTextAnnotations.stream().limit(3).collect(Collectors.toList());
                }

                // 선택된 텍스트 블록을 왼쪽에서 오른쪽, 상단에서 하단으로 정렬
                List<EntityAnnotation> sortedSelectedAnnotations = selectedAnnotations.stream()
                        .sorted(Comparator.comparingDouble(this::getBoundingBoxLeft)
                                .thenComparingDouble(this::getBoundingBoxTop))
                        .collect(Collectors.toList());

                // 중복된 단어 제거 및 조합
                Set<String> uniqueWords = new LinkedHashSet<>();
                for (EntityAnnotation annotation : sortedSelectedAnnotations) {
                    String[] words = removeSpecialCharacters(annotation.getDescription()).split("\\s+");
                    Collections.addAll(uniqueWords, words);
                }
                finalText = new ArrayList<>(uniqueWords);
            }

            // 결과를 하나의 문자열로 조합
            return String.join(" ", finalText);
        } catch (IOException e) {
            e.printStackTrace();
            return "Error occurred while processing the image.";
        }
    }

    // 경계 상자의 면적 계산
    private double calculateBoundingBoxArea(EntityAnnotation annotation) {
        BoundingPoly boundingPoly = annotation.getBoundingPoly();
        if (boundingPoly.getVerticesCount() < 4) {
            return 0;
        }
        Rectangle2D rect = boundingBoxToRectangle(boundingPoly);
        return rect.getWidth() * rect.getHeight();
    }

    // 경계 상자를 Rectangle2D 객체로 변환
    private Rectangle2D boundingBoxToRectangle(BoundingPoly boundingPoly) {
        List<Vertex> vertices = boundingPoly.getVerticesList();
        return new Rectangle2D.Double(
                vertices.get(0).getX(),
                vertices.get(0).getY(),
                vertices.get(2).getX() - vertices.get(0).getX(),
                vertices.get(2).getY() - vertices.get(0).getY()
        );
    }

    // 경계 상자의 왼쪽 위치 가져오기
    private double getBoundingBoxLeft(EntityAnnotation annotation) {
        BoundingPoly boundingPoly = annotation.getBoundingPoly();
        List<Vertex> vertices = boundingPoly.getVerticesList();
        return vertices.stream().mapToDouble(Vertex::getX).min().orElse(0);
    }

    // 경계 상자의 상단 위치 가져오기
    private double getBoundingBoxTop(EntityAnnotation annotation) {
        BoundingPoly boundingPoly = annotation.getBoundingPoly();
        List<Vertex> vertices = boundingPoly.getVerticesList();
        return vertices.stream().mapToDouble(Vertex::getY).min().orElse(0);
    }

    // 특수 기호 제거 (알파벳, 숫자, 한글 및 공백 유지)
    private String removeSpecialCharacters(String text) {
        return text.replaceAll("[^a-zA-Z0-9가-힣\\s]", "").trim();
    }

    // 한글과 숫자만 추출
    private String extractHangulAndNumbers(String text) {
        return text.replaceAll("[^가-힣0-9\\s]", "").trim();
    }
}
