import 'package:flutter/material.dart';
 
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
 
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}
 
class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _fade;
 
  @override
  void initState() {
    super.initState();
 
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
 
    _fade = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeIn,
    );
 
    _controller.forward();
 
    Future.delayed(const Duration(seconds: 5), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacementNamed('/main_layout');
    });
  }
 
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
 
  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
 
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: FadeTransition(
          opacity: _fade,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Image.asset(
              'images/convo_docs_logo.png',
              width: screenWidth * 0.72, // 72% of screen width — looks great on all sizes
              fit: BoxFit.contain,
            ),
          ),
        ),
      ),
    );
  }
}
 