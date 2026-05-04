import 'dart:math';

// Represents a face embedding — a list of 512 numbers that
// uniquely describe a face (output of MobileFaceNet)
// Think of it like a fingerprint, but as numbers

class FaceEmbedding {
  final List<double> values;

  const FaceEmbedding(this.values);

  // Convert to JSON for storage
  Map<String, dynamic> toJson() => {'values': values};

  // Load from JSON storage
  factory FaceEmbedding.fromJson(Map<String, dynamic> json) {
    return FaceEmbedding(List<double>.from(json['values']));
  }

  // Cosine similarity — measures how similar two embeddings are
  // Returns a value between -1 and 1, where 1 = identical face
  // This is what FR-10 requires
  double cosineSimilarity(FaceEmbedding other) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < values.length; i++) {
      dotProduct += values[i] * other.values[i];
      normA += values[i] * values[i];
      normB += other.values[i] * other.values[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0;

    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  // Average multiple embeddings together for enrolment
  // We capture 3 frames and average them for a more stable profile
  static FaceEmbedding average(List<FaceEmbedding> embeddings) {
    final length = embeddings.first.values.length;
    final averaged = List<double>.filled(length, 0.0);

    for (final embedding in embeddings) {
      for (int i = 0; i < length; i++) {
        averaged[i] += embedding.values[i];
      }
    }

    for (int i = 0; i < length; i++) {
      averaged[i] /= embeddings.length;
    }

    return FaceEmbedding(averaged);
  }
}