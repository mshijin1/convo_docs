import 'package:flutter/material.dart';

class Sidebar extends StatelessWidget {
  const Sidebar({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 250,
      decoration: const BoxDecoration(
        color: Color(0xFFFAFAFA), // Light grey sidebar background
        border: Border(
          right: BorderSide(
            color: Color(0xFFE5E7EB), // Divider line on the right
            width: 1.0,
          ),
        ),
      ),
      child: const Center(
        child: Text(
          'Sidebar Items Go Here',
          style: TextStyle(
            color: Color(0xFF9CA3AF),
            fontSize: 13,
          ),
        ),
      ),
    );
  }
}