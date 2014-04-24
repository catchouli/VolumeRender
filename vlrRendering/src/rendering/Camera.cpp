#include "rendering/Camera.h"

#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vlr
{
	namespace rendering
	{
		Camera::Camera()
		{

		}
		
		Camera::Camera(const glm::vec3& position, const glm::quat& rotation)
			: _position(position), _rotation(rotation)
		{
			setViewport(0, 0, 0, 0);
		}

		Camera::~Camera()
		{

		}

		void Camera::updateGL()
		{
			// Update view matrix
			_rotationMatrix = glm::toMat4(_rotation);
			_view = glm::translate(_rotationMatrix, -(_position));

			// Calculate mvp
			glm::mat4 mvp = getMVP();

			// Set viewport
			glViewport(_viewport.x, _viewport.y, _viewport.w, _viewport.h);

			// Set projection
			glMatrixMode(GL_PROJECTION);
			glLoadMatrixf(glm::value_ptr(mvp));

			// Initialise modelview
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
		}

		void Camera::setViewport(int x, int y, int width, int height)
		{
			_viewport.x = x;
			_viewport.y = y;
			_viewport.w = width;
			_viewport.h = height;
		}

		void Camera::perspective(float fov, float aspect,
			float near, float far)
		{
			_fov = fov;
			_aspect = aspect;
			_near = near;
			_far = far;

			_projection = glm::perspective(fov, aspect, near, far);
		}

		void Camera::orthographic(float scale, float aspect)
		{
			_projection = glm::ortho(scale * -1.0f, scale * 1.0f,
				(scale * -1.0f) / aspect, (scale * 1.0f) / aspect);
		}


		void Camera::setPos(const glm::vec3& pos)
		{
			_position = pos;
		}

		void Camera::setRot(const glm::quat quat)
		{
			_rotation = quat;
		}

		void Camera::translate(const glm::vec3& offset)
		{
			_position += offset;
		}

		void Camera::rotate(const glm::quat& quat)
		{
			_rotation *= quat;
		}

		void Camera::rotate(const glm::vec3& euler)
		{
			_rotation *= glm::quat(euler);
		}

		glm::vec3 Camera::screenSpaceToWorld(float x, float y,
			float dist) const
		{
			// Calculate viewport space of cursor
			double viewportX = 2.0 * ((x - _viewport.x) / (double)_viewport.w) - 1.0;
			double viewportY = 2.0 * ((y - _viewport.y) / (double)_viewport.h) - 1.0;
			//double viewportY = (2.0 - 2.0 * ((y - _viewport.y) / (double)_viewport.h)) + 1.0;

			// Get camera matrix
			glm::mat4 mat = getMVP();
			glm::mat4 invmat = glm::inverse(mat);

			// Convert viewport cursor pos to world
			glm::vec4 viewportPos(viewportX, viewportY, 1, 1);
			glm::vec4 worldPos = viewportPos * invmat + glm::vec4(_position, 0);

			if (_isnan(worldPos.x))
			{
				int i = 0;
			}

			return glm::vec3(worldPos);
		}

		glm::vec3 Camera::worldSpaceToScreen(float x, float y, float z) const
		{
			glm::vec4 worldPos(x, y, z, 0);
			glm::vec4 viewportPos = getMVP() * (worldPos - glm::vec4(_position, 0));
			
			double screenX = 0.5f * (viewportPos.x + 1.0f) * _viewport.w + _viewport.x;
			double screenY = 0.5f * (viewportPos.y + 1.0f) * _viewport.h + _viewport.y;

			return glm::vec3(screenX, screenY, 0);
		}
	}
}
