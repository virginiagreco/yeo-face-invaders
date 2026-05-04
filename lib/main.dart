import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'enrolment/enrolment_screen.dart';
import 'game/game_screen.dart';

void main() async {
  // Ensure Flutter is initialised before accessing camera
  WidgetsFlutterBinding.ensureInitialized();

  // Get available cameras
  final cameras = await availableCameras();

  runApp(YeoFaceInvadersApp(cameras: cameras));
}

class YeoFaceInvadersApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const YeoFaceInvadersApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YEO Face Invaders',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      initialRoute: '/enrolment',
      routes: {
        '/enrolment': (context) => EnrolmentScreen(cameras: cameras),
        '/game': (context) => GameScreen(cameras: cameras),
      },
    );
  }
}