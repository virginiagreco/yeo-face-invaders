import 'package:flutter/material.dart';

// Game constants
const double kPlayerSpeed = 200.0;
const double kBulletSpeed = 400.0;
const double kInvaderSpeed = 40.0;
const double kInvaderDropDistance = 20.0;

class Player {
  double x;
  double y;
  final double width = 40;
  final double height = 30;

  Player({required this.x, required this.y});

  Rect get rect => Rect.fromLTWH(x - width / 2, y - height / 2, width, height);
}

class Bullet {
  double x;
  double y;
  final double width = 4;
  final double height = 12;
  bool active = true;

  Bullet({required this.x, required this.y});

  Rect get rect => Rect.fromLTWH(x - width / 2, y - height / 2, width, height);
}

class Invader {
  double x;
  double y;
  final double width = 30;
  final double height = 24;
  bool alive = true;
  int row;

  Invader({required this.x, required this.y, required this.row});

  Rect get rect => Rect.fromLTWH(x - width / 2, y - height / 2, width, height);
}

class SpaceInvadersGame {
  // Game dimensions — set when game starts
  double screenWidth = 0;
  double screenHeight = 0;

  late Player player;
  List<Bullet> bullets = [];
  List<Invader> invaders = [];

  int score = 0;
  bool gameOver = false;
  bool playerWon = false;

  // Invader movement
  double _invaderDirection = 1.0; // 1 = right, -1 = left
  double _invaderMoveTimer = 0.0;
  static const double _invaderMoveInterval = 0.8;

  // Shooting cooldown
  double _shootCooldown = 0.0;
  static const double _shootCooldownDuration = 0.4;

  void initialise(double width, double height) {
    screenWidth = width;
    screenHeight = height;

    // Place player at bottom centre
    player = Player(x: width / 2, y: height - 60);

    // Create invader grid — 5 rows x 8 columns (FR-17)
    invaders.clear();
    for (int row = 0; row < 5; row++) {
      for (int col = 0; col < 8; col++) {
        invaders.add(Invader(
          x: 50.0 + col * 45.0,
          y: 80.0 + row * 40.0,
          row: row,
        ));
      }
    }

    score = 0;
    gameOver = false;
    playerWon = false;
    bullets.clear();
  }

  // Main game update — called every frame (FR-22)
  void update(double dt) {
    if (gameOver) return;

    _updateBullets(dt);
    _updateInvaders(dt);
    _checkCollisions();
    _checkGameOver();

    if (_shootCooldown > 0) _shootCooldown -= dt;
  }

  void _updateBullets(double dt) {
    for (final bullet in bullets) {
      if (bullet.active) bullet.y -= kBulletSpeed * dt;
    }
    // Remove off-screen bullets
    bullets.removeWhere((b) => !b.active || b.y < 0);
  }

  void _updateInvaders(double dt) {
    _invaderMoveTimer += dt;
    if (_invaderMoveTimer < _invaderMoveInterval) return;
    _invaderMoveTimer = 0;

    final aliveInvaders = invaders.where((i) => i.alive).toList();
    if (aliveInvaders.isEmpty) return;

    // Check if any invader hits the edge
    bool hitEdge = false;
    for (final invader in aliveInvaders) {
      if ((invader.x + invader.width / 2 >= screenWidth - 10 &&
              _invaderDirection > 0) ||
          (invader.x - invader.width / 2 <= 10 && _invaderDirection < 0)) {
        hitEdge = true;
        break;
      }
    }

    if (hitEdge) {
      // Drop down and reverse direction
      _invaderDirection *= -1;
      for (final invader in aliveInvaders) {
        invader.y += kInvaderDropDistance;
      }
    } else {
      // Move horizontally
      for (final invader in aliveInvaders) {
        invader.x += kInvaderSpeed * _invaderDirection * _invaderMoveInterval;
      }
    }
  }

  // Collision detection (FR-18)
  void _checkCollisions() {
    for (final bullet in bullets) {
      if (!bullet.active) continue;
      for (final invader in invaders) {
        if (!invader.alive) continue;
        if (bullet.rect.overlaps(invader.rect)) {
          bullet.active = false;
          invader.alive = false;
          score += 10;
          break;
        }
      }
    }
  }

  void _checkGameOver() {
    // Player wins if all invaders destroyed
    if (invaders.every((i) => !i.alive)) {
      gameOver = true;
      playerWon = true;
      return;
    }

    // Player loses if invaders reach the bottom
    for (final invader in invaders) {
      if (invader.alive && invader.y >= screenHeight - 80) {
        gameOver = true;
        return;
      }
    }
  }

  // Move player left or right
  void movePlayer(double dx, double dt) {
    player.x += dx * kPlayerSpeed * dt;
    player.x = player.x.clamp(player.width / 2, screenWidth - player.width / 2);
  }

  // Fire a bullet (FR-16)
  void shoot() {
    if (_shootCooldown > 0) return;
    bullets.add(Bullet(x: player.x, y: player.y - player.height / 2));
    _shootCooldown = _shootCooldownDuration;
  }

  // Reset for new game
  void reset() {
    initialise(screenWidth, screenHeight);
  }
}

// Custom painter — draws the game every frame (FR-20)
class SpaceInvadersPainter extends CustomPainter {
  final SpaceInvadersGame game;

  SpaceInvadersPainter(this.game);

  @override
  void paint(Canvas canvas, Size size) {
    // Black background
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint()..color = Colors.black,
    );

    // Draw player ship
    _drawPlayer(canvas, game.player);

    // Draw bullets
    final bulletPaint = Paint()..color = Colors.greenAccent;
    for (final bullet in game.bullets) {
      if (bullet.active) {
        canvas.drawRect(bullet.rect, bulletPaint);
      }
    }

    // Draw invaders
    for (final invader in game.invaders) {
      if (invader.alive) {
        _drawInvader(canvas, invader);
      }
    }

    // Draw score (FR-19)
    final textPainter = TextPainter(
      text: TextSpan(
        text: 'SCORE: ${game.score}',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 18,
          fontFamily: 'monospace',
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, const Offset(16, 16));

    // Game over overlay
    if (game.gameOver) {
      _drawGameOver(canvas, size);
    }
  }

  void _drawPlayer(Canvas canvas, Player player) {
    final paint = Paint()..color = Colors.cyanAccent;
    final path = Path();

    // Simple triangle ship shape
    path.moveTo(player.x, player.y - player.height / 2);
    path.lineTo(player.x - player.width / 2, player.y + player.height / 2);
    path.lineTo(player.x + player.width / 2, player.y + player.height / 2);
    path.close();

    canvas.drawPath(path, paint);
  }

  void _drawInvader(Canvas canvas, Invader invader) {
    // Different colours per row
    final colors = [
      Colors.redAccent,
      Colors.orangeAccent,
      Colors.yellowAccent,
      Colors.greenAccent,
      Colors.purpleAccent,
    ];
    final paint = Paint()..color = colors[invader.row % colors.length];

    // Simple invader body
    canvas.drawRect(invader.rect, paint);

    // Invader "eyes"
    final eyePaint = Paint()..color = Colors.black;
    canvas.drawCircle(
      Offset(invader.x - 6, invader.y - 4),
      3,
      eyePaint,
    );
    canvas.drawCircle(
      Offset(invader.x + 6, invader.y - 4),
      3,
      eyePaint,
    );
  }

  void _drawGameOver(Canvas canvas, Size size) {
    final overlay = Paint()..color = Colors.black.withValues(alpha: 0.7);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), overlay);

    final text = game.playerWon ? 'YOU WIN!' : 'GAME OVER';
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: game.playerWon ? Colors.greenAccent : Colors.redAccent,
          fontSize: 48,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        size.width / 2 - textPainter.width / 2,
        size.height / 2 - textPainter.height / 2,
      ),
    );
  }

  @override
  bool shouldRepaint(SpaceInvadersPainter oldDelegate) => true;
}